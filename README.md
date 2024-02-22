## Optimized Masked Autoencoders (MAEs)

An optimized implementation of masked autoencoders (MAEs). The following optimizations are planned to be implemented:

- [x] FlashAttention-2
- [x] `torch.compile`
- [x] optimized AdamW (`foreach` and `fused`)
- [ ] `FSDP` for distributed training