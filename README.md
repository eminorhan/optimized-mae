## Optimized Masked Autoencoders (MAEs)

A lean, optimized implementation of masked autoencoders (MAEs). The skeleton of the code is recycled from Facebook's [MAE](https://github.com/facebookresearch/mae) repository with various simplifications. The following optimizations are implemented:

- [x] FlashAttention-2
- [x] `torch.compile`
- [x] `fused` AdamW
- [x] mixed precision training (`torch.cuda.amp`)
- [x] `DDP` for distributed training
- [x] (optional) [`webdataset`](https://github.com/webdataset/webdataset) based data loader

Dependence of model definitions on the `timm` library is also removed in this implementation, so the code is self-contained except for the standard libraries. The code was tested with `pytorch==2.2.0` and `torchvision==0.17.0`.

**Notes:**

- **Pretraining (`train.py`):** On a single A100 (with 80 GB GPU memory), it is currently possible to train an MAE with a ViT-H/14 encoder on 1792x1792 images! With 80% masking ratio, you can fit a minibatch size of 8 on a single GPU in this configuration. `train_wds.py` provides a faster version of the pretraining code with a `webdataset` based data loader. 
- **Finetuning (`finetune.py`):** The same model can then be finetuned on a downstream recognition task (again with 1792x1792 images). A single 80GB A100 can fit a batch size of 2 in this setting (this is smaller than in pretraining, because there is no masking in this case).
- The code currently uses DDP for distributed training and it can be scaled up painlessly to multinode training setups without any modifications. See the [`scripts`](https://github.com/eminorhan/optimized-mae/tree/master/scripts) folder for example distributed training scripts for a SLURM cluster.
- `torch.compile` yields marginal improvements in my experience (typically around 3-4% improvement in pretraining runtime).

