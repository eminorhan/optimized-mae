## Optimized Masked Autoencoders (MAEs)

An optimized, lean implementation of masked autoencoders (MAEs). The following optimizations are planned to be implemented:

- [x] FlashAttention-2
- [x] `torch.compile`
- [x] optimized AdamW (`foreach` and `fused`)
- [ ] `FSDP` for distributed training

Dependence of model definitions on the `timm` library is also removed in this implementations, so the code is self-contained except for the standard libraries.