"""Explicit prompt coordinates, independent of PEFT version behavior."""
import torch
from torch import nn


class PromptedQA(nn.Module):
    def __init__(self, base, prompt_length=0):
        super().__init__()
        self.base = base
        self.prompt_length = prompt_length
        if prompt_length:
            for p in self.base.parameters():
                p.requires_grad_(False)
            embeddings = base.get_input_embeddings().weight
            self.prompt = nn.Parameter(embeddings[:prompt_length].detach().clone())
            # Train QA head alongside the prompt, as in the original experiment.
            for p in self.base.qa_outputs.parameters():
                p.requires_grad_(True)

    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        p = self.prompt_length
        kwargs = {"attention_mask": attention_mask}
        if p:
            batch = input_ids.shape[0]
            kwargs["inputs_embeds"] = torch.cat([self.prompt.unsqueeze(0).expand(batch, -1, -1), self.base.get_input_embeddings()(input_ids)], dim=1)
            kwargs["attention_mask"] = torch.cat([torch.ones((batch, p), dtype=attention_mask.dtype, device=attention_mask.device), attention_mask], dim=1)
        else:
            kwargs["input_ids"] = input_ids
        if start_positions is not None:
            if end_positions is None or torch.any(start_positions < 0) or torch.any(end_positions < start_positions) or torch.any(end_positions >= input_ids.shape[1]):
                raise ValueError("Invalid QA labels")
            kwargs["start_positions"] = start_positions + p
            kwargs["end_positions"] = end_positions + p
        output = self.base(**kwargs)
        # Evaluation offsets stay in tokenizer coordinates, including CLS.
        return output.loss, output.start_logits[:, p:], output.end_logits[:, p:]
