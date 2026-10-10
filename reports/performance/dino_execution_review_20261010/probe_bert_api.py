"""Probe the installed BERT API; this does not load LAE-DINO or model weights."""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path


def main() -> None:
    import torch
    from transformers import BertConfig, BertModel

    result = {
        "scope": "tiny randomly initialized CPU BERT; no detector/checkpoint/GPU benchmark",
        "python": sys.executable,
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "cases": [],
    }
    # LAE's sub-sentence mask is [batch, tokens, tokens], not the usual 2D mask.
    # Test both the library default and explicit eager attention without patching LAE.
    for implementation in (None, "eager"):
        config = BertConfig(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
        )
        if implementation:
            config._attn_implementation = implementation
        model = BertModel(config, add_pooling_layer=False).eval()
        for batch_size in (1, 2):
            for dimensions in (2, 3):
                entry = {
                    "requested_attention": implementation or "default",
                    "resolved_attention": model.config._attn_implementation,
                    "batch_size": batch_size,
                    "mask_dimensions": dimensions,
                }
                ids = torch.ones(batch_size, 8, dtype=torch.long)
                mask = torch.ones(
                    (batch_size, 8, 8) if dimensions == 3 else (batch_size, 8),
                    dtype=torch.bool,
                )
                try:
                    with torch.no_grad():
                        prediction = model(
                            input_ids=ids,
                            attention_mask=mask,
                            position_ids=torch.arange(8).expand(batch_size, -1),
                            token_type_ids=torch.zeros_like(ids),
                            output_hidden_states=True,
                        )
                    entry.update(status="ok", output_shape=list(prediction.last_hidden_state.shape))
                except Exception as exc:
                    entry.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                result["cases"].append(entry)
    path = Path(__file__).with_name("bert_api_probe.json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
