"""Data-independent QA contracts and span post-processing."""
import hashlib
import json
import math
from pathlib import Path


def validate_examples(examples):
    if not examples:
        raise ValueError("At least one example is required")
    seen = set()
    for ex in examples:
        ident = ex.get("id")
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError("Each example must have a unique nonempty string id")
        seen.add(ident)
        answers = ex["answers"]
        if len(answers["text"]) != len(answers["answer_start"]):
            raise ValueError("Answer text/start count mismatch")
        for text, start in zip(answers["text"], answers["answer_start"]):
            if not isinstance(start, int) or start < 0 or ex["context"][start:start + len(text)] != text:
                raise ValueError("Answer must match original character coordinates")
    return examples


def span_labels(offsets, answer_start, answer_text, cls_index=0):
    context = [i for i, off in enumerate(offsets) if off is not None]
    end = answer_start + len(answer_text)
    if not context or offsets[context[0]][0] > answer_start or offsets[context[-1]][1] < end:
        return cls_index, cls_index
    starts = [i for i in context if offsets[i][0] <= answer_start < offsets[i][1]]
    ends = [i for i in context if offsets[i][0] < end <= offsets[i][1]]
    if not starts or not ends:
        raise ValueError("Answer boundary cannot be aligned to context tokens")
    return starts[0], ends[-1]


def shift_label(label, prompt_length, token_length):
    if not 0 <= label < token_length or prompt_length < 0:
        raise ValueError("Invalid token coordinate")
    return label + prompt_length


def best_spans(examples, features, starts, ends, max_answer_length=30):
    validate_examples(examples)
    if not len(features) == len(starts) == len(ends):
        raise ValueError("Feature/logit count mismatch")
    by_id = {ex["id"]: ex for ex in examples}
    candidates = {}
    for feature, start, end in zip(features, starts, ends):
        ident = feature["example_id"]
        if ident not in by_id:
            raise ValueError("Unknown feature example id")
        offsets = feature["offset_mapping"]
        if len(offsets) != len(start) or len(offsets) != len(end):
            raise ValueError("Logits must be in original token coordinates")
        for i, off in enumerate(offsets):
            if off is None:
                continue
            for j in range(i, min(i + max_answer_length, len(offsets))):
                if offsets[j] is None:
                    break
                score = float(start[i]) + float(end[j])
                if not math.isfinite(score):
                    raise ValueError("Non-finite logits")
                if ident not in candidates or score > candidates[ident][0]:
                    text = by_id[ident]["context"][off[0]:offsets[j][1]]
                    candidates[ident] = score, text
    if set(candidates) != set(by_id):
        raise ValueError("Every example must have at least one context feature")
    return [{"id": ex["id"], "prediction_text": candidates[ex["id"]][1]} for ex in examples]


def write_checkpoint_manifest(checkpoint, metadata):
    checkpoint = Path(checkpoint)
    manifest = {**metadata, "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    target = checkpoint.with_suffix(checkpoint.suffix + ".json")
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    temp.replace(target)


def verify_checkpoint(checkpoint, expected_prompt_length):
    checkpoint = Path(checkpoint)
    meta = json.loads(checkpoint.with_suffix(checkpoint.suffix + ".json").read_text(encoding="utf-8"))
    if meta["prompt_length"] != expected_prompt_length:
        raise ValueError("Checkpoint/model prompt length mismatch")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("Checkpoint integrity mismatch")
    return meta
