# Checkpoints

| Model | File | R2R-CE val-unseen SR | SPL | nDTW | Download |
|---|---|---:|---:|---:|---|
| Safe-VLN, iter12800 | `ckpt.iter12800.pth` | 60.25 | 48.45 | 59.59 | [Aliyun Drive](https://www.alipan.com/s/VAvCoGBxhDJ) |

Download the checkpoint from the shared folder. No extraction code is required;
the share link is set to never expire. The folder also contains the SHA256 file,
model metadata and a short README.

## File verification

- Size: **945,159,739 bytes** (about 901 MiB).
- SHA256: `679012e80677f548c61e534809b416906e30cd933982ccc984514c44f28a1453`.
- Contents: model tensors, iteration, and checkpoint format version.
- Optimizer state and machine-specific configuration are not included.

```bash
sha256sum data/checkpoints/ckpt.iter12800.pth
```

The exported tensor names, values and dtypes match the selected training
checkpoint exactly. It is intended for evaluation or initializing a new
fine-tuning run, not for resuming the original optimizer state.

Keep `EVAL.USE_CKPT_CONFIG=False` and `IL.is_requeue=False`, as set by the
provided launcher. ETP initialization, waypoint predictor, depth encoder,
CLIP weights and scene data are still required; see the root README.

Machine-readable metadata: [checkpoint.json](../results/checkpoint.json).
