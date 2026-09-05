# RS Merger Expert experiment matrix

Runtime values are populated after formal C0/C1/C2/C3 runs. Local execution is currently blocked
before real-model smoke because the exact Qwen3-VL-4B/R1 assets are absent.

| experiment | architecture | trainable params | VRAM | runtime | exact | within 1 | MAE | bias | 6-10 exact | 6-10 within 1 | 6-10 MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | formal R1 base route | 0 | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| C1 | four cloned R1 mergers | 109,105,152 under expected 4B contract | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| C2 | four independent zero-init RS detail branches | 24,160,256 if 1024/2560/2 contract holds | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| C3 | C2 + exact LLM 0-3 q/k/v/o LoRA | 25,470,976 under expected 4B contract | pending | pending | pending | pending | pending | pending | pending | pending | pending |
