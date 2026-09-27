"""Train/evaluate a local, authorized SQuAD-style JSON dataset."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from modeling import PromptedQA
from qa_core import validate_examples, span_labels, best_spans, write_checkpoint_manifest, verify_checkpoint


def features_for(examples, tokenizer, max_length):
    validate_examples(examples)
    result = []
    for ex in examples:
        encoded = tokenizer(ex["question"], ex["context"], truncation="only_second", max_length=max_length, stride=min(128, max_length // 4), return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length")
        for i, ids in enumerate(encoded["input_ids"]):
            offsets = [off if seq == 1 else None for off, seq in zip(encoded["offset_mapping"][i], encoded.sequence_ids(i))]
            cls = ids.index(tokenizer.cls_token_id)
            answers = ex["answers"]
            start, end = span_labels(offsets, answers["answer_start"][0], answers["text"][0], cls) if answers["text"] else (cls, cls)
            result.append({"input_ids": ids, "attention_mask": encoded["attention_mask"][i], "start_positions":start, "end_positions":end, "example_id":ex["id"], "offset_mapping":offsets})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "evaluate"])
    parser.add_argument("--data", required=True, help="Local JSON list with id/question/context/answers")
    parser.add_argument("--model", default="microsoft/deberta-v3-base")
    parser.add_argument("--revision", required=True, help="Pin the pretrained model/tokenizer revision")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-length", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.prompt_length < 256:
        parser.error("prompt-length must be in [0, 256)")
    torch.manual_seed(args.seed)
    examples = json.loads(Path(args.data).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    feats = features_for(examples, tokenizer, 512 - args.prompt_length)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PromptedQA(AutoModelForQuestionAnswering.from_pretrained(args.model, revision=args.revision), args.prompt_length).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    checkpoint = Path(args.checkpoint)
    start_epoch = 0
    if args.mode == "evaluate" or args.resume:
        meta = verify_checkpoint(checkpoint, args.prompt_length)
        if meta["model"] != args.model or meta["revision"] != args.revision:
            raise ValueError("Checkpoint pretrained model/revision mismatch")
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(saved["model"], strict=True)
        if args.resume:
            optimizer.load_state_dict(saved["optimizer"])
            start_epoch = saved["epoch"]
    elif checkpoint.exists():
        raise FileExistsError("Use --resume or a new checkpoint path")
    keys = ["input_ids", "attention_mask", "start_positions", "end_positions"]
    rows = [{k:torch.tensor(f[k]) for k in keys} for f in feats]
    if args.mode == "train":
        for epoch in range(start_epoch, args.epochs):
            model.train()
            for batch in DataLoader(rows, batch_size=args.batch_size, shuffle=True):
                optimizer.zero_grad()
                loss, _, _ = model(**{k:v.to(device) for k,v in batch.items()})
                loss.backward()
                optimizer.step()
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temp = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
            torch.save({"model":model.state_dict(), "optimizer":optimizer.state_dict(), "epoch":epoch+1}, temp)
            temp.replace(checkpoint)
            write_checkpoint_manifest(checkpoint, {"prompt_length":args.prompt_length, "model":args.model, "revision":args.revision, "epoch":epoch+1, "seed":args.seed})
        print("Checkpoint saved; evaluation is a separate explicit mode.")
    else:
        starts, ends = [], []
        model.eval()
        with torch.no_grad():
            for batch in DataLoader(rows, batch_size=args.batch_size):
                _, s, e = model(**{k:v.to(device) for k,v in batch.items() if k in ["input_ids", "attention_mask"]})
                starts.extend(s.cpu().tolist()); ends.extend(e.cpu().tolist())
        predictions = best_spans(examples, feats, starts, ends)
        # Do not print clinical text. Publish only aggregate numbers from authorized evaluation.
        exact = sum(p["prediction_text"] in ex["answers"]["text"] for p,ex in zip(predictions,examples))
        print(json.dumps({"examples":len(examples), "exact_string_match":exact / len(examples)}))


if __name__ == "__main__":
    main()
