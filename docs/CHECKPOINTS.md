# Checkpoints

| Model | File | R2R-CE val-unseen SR | SPL | nDTW | Download |
|---|---|---:|---:|---:|---|
| Safe-VLN, iter12800 | `Safe-VLN-iter12800.zip` | 60.25 | 48.45 | 59.59 | [GitHub Releases](https://github.com/yueluhhxx/Safe-VLN/releases/download/checkpoint-iter12800/Safe-VLN-iter12800.zip) |

Download `Safe-VLN-iter12800.zip` from GitHub Releases and extract it.
The archive contains the checkpoint, its SHA256 file, model metadata and a short README.

An [Aliyun Drive fast-transfer mirror](https://www.aliyundrive.com/t/HlRhz8xSxnX11rjTMVpb)
is available until **2026-11-09 22:41 (UTC+8)**. It requires no extraction code.
Use GitHub Releases after the mirror expires.

Archive SHA256: `8f6f42946c7676e6d7fcfdf4169c296b41afd14202418a95543c6b4f57b8675e`.

## File verification

The following values refer to the extracted `ckpt.iter12800.pth`:

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
