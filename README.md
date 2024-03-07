## Optimized Masked Autoencoders (MAEs)

A lean, optimized implementation of masked autoencoders (MAEs). The skeleton of the code is recycled from Facebook's [MAE](https://github.com/facebookresearch/mae) repository with various simplifications. The following optimizations are implemented:

- [x] FlashAttention-2
- [x] `torch.compile`
- [x] `fused` AdamW
- [ ] `FSDP` for distributed training

Dependence of model definitions on the `timm` library is also removed in this implementation, so the code is self-contained except for the standard libraries.

**Notes:**

- On a single A100 (with 80 GB GPU memory), it is currently possible to train an MAE with a ViT-H/14 encoder on 1792x1792 images! With 80% masking ratio, you can fit a minibatch size of 8 on a single GPU in this configuration.
- `torch.compile` yields marginal improvements in my experience (typically around 3-4% improvement in pretraining runtime).
- This project is very much work in progress at the moment.

