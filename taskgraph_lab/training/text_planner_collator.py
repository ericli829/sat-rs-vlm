"""Assistant-only text collator for a Qwen3 Planner."""

from __future__ import annotations

from typing import Any


class PlannerCausalLMCollator:
    """Render no-thinking chat prompts and supervise only the Planner DSL."""

    def __init__(self, tokenizer: Any, *, max_seq_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def _render(self, messages: list[dict[str, Any]], *, prompt: bool) -> str:
        selected = (
            [message for message in messages if message.get("role") != "assistant"]
            if prompt
            else messages
        )
        return str(
            self.tokenizer.apply_chat_template(
                selected,
                tokenize=False,
                add_generation_prompt=prompt,
                enable_thinking=False,
            )
        )

    def _encode(self, texts: list[str]) -> dict[str, Any]:
        return dict(
            self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
        )

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        full = self._encode([self._render(row["messages"], prompt=False) for row in batch])
        prompts = self._encode([self._render(row["messages"], prompt=True) for row in batch])
        labels = full["input_ids"].clone()
        full_mask = full["attention_mask"]
        prompt_mask = prompts["attention_mask"]
        for index, row in enumerate(batch):
            full_length = int(full_mask[index].sum().item())
            prompt_length = int(prompt_mask[index].sum().item())
            if full_length <= prompt_length:
                raise ValueError(f"{row.get('id')}: no assistant tokens remain after tokenization")
            labels[index, :prompt_length] = -100
            labels[index, full_length:] = -100
            labels[index, full_mask[index] == 0] = -100
        full["labels"] = labels
        return full

    def diagnostics(self, row: dict[str, Any]) -> dict[str, int | bool]:
        full_text = self._render(row["messages"], prompt=False)
        prompt_text = self._render(row["messages"], prompt=True)
        full_uncapped = self.tokenizer(full_text, truncation=False)
        prompt_uncapped = self.tokenizer(prompt_text, truncation=False)
        encoded = self([row])
        total = int(encoded["attention_mask"][0].sum().item())
        supervised = int((encoded["labels"][0] != -100).sum().item())
        uncapped_total = len(full_uncapped["input_ids"])
        return {
            "prompt_tokens": total - supervised,
            "assistant_tokens": supervised,
            "total_tokens": total,
            "uncapped_prompt_tokens": len(prompt_uncapped["input_ids"]),
            "uncapped_total_tokens": uncapped_total,
            "truncated": uncapped_total > total,
        }

