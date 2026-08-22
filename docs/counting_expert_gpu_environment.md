# Counting Expert GPU environment gate

Run the gate before any cloud smoke or formal experiment:

```bash
python scripts/environment/check_gpu_environment.py --strict-5090 --attention-backend auto
```

The strict 5090 gate requires an RTX 5090 reporting compute capability 12.0, a PyTorch binary
containing `sm_120`, BF16, and SDPA. The target stack is PyTorch 2.12 + CUDA 13; the minimum
fallback is PyTorch 2.7 + CUDA 12.8 with an actual `sm_120` binary.

`auto` always has SDPA as the safe baseline. It selects FlashAttention 2 only after import and a
real BF16 CUDA forward/backward finite-value smoke. Use `--attention-backend sdpa` to force the
baseline, or `flash_attention_2` to fail closed when the FlashAttention smoke does not pass.

For memory isolation, launch every formal variant through
`scripts/training/run_rs_merger_experiments.py`. Each variant is a separate subprocess; the parent
waits for its exit before starting the next one, so a 4B CUDA context never survives into the next
experiment.

The local RTX 4060 report is diagnostic only. It does not satisfy the strict RTX 5090 cloud gate.
