<div align="center">

# Safe-VLN

### Collision Avoidance for Vision-and-Language Navigation<br>in Continuous Environments

[Paper](https://arxiv.org/abs/2311.02817) · [Installation](docs/INSTALL.md) · [Data](#data) · [Checkpoints](docs/CHECKPOINTS.md)

<img src="https://arxiv.org/html/2311.02817v2/main_idea1.png" width="1000" alt="Safe-VLN overview: occupancy-aware waypoint prediction and re-selection navigation">

*Safe-VLN overview, Figure 2 of the [paper](https://arxiv.org/html/2311.02817v2#S3).*

</div>

Safe-VLN combines occupancy-aware waypoint prediction with waypoint re-selection
after collisions. The implementation builds on [ETPNav](https://github.com/MarSaKi/ETPNav)
and follows the planar depth-sensing setup in [Sim2Sim-VLNCE](https://github.com/jacobkrantz/Sim2Sim-VLNCE).

## Installation

```bash
git clone https://github.com/yueluhhxx/Safe-VLN.git
cd Safe-VLN
```

Follow [the installation guide](docs/INSTALL.md) to create the environment and
install Habitat-Lab, Habitat-Sim and CLIP. The recorded setup uses Linux,
Python 3.8, PyTorch 2.0.1 + CUDA 11.7, and Habitat-Sim 0.1.7.
The training and evaluation commands below use **8 GPUs with 2 environments per GPU**.

## Data

Download the R2R-CE training/validation episodes and pretrained assets from
[ETPNav on ModelScope](https://modelscope.cn/models/admagic/ETPNav/files).
This code uses the **R2R_VLNCE_v1-2_preprocessed_BERTidx** episodes and the
corresponding **R2R_VLNCE_v1-2_preprocessed** reference paths.

The upstream download command is:

```bash
modelscope download --model admagic/ETPNav --local_dir ./data
```

Use a separate environment for the download CLI if needed; do not upgrade the
pinned navigation environment to install download tools.

Matterport3D scene meshes require access from the
[Matterport3D project](https://niessner.github.io/Matterport/).
See the [VLN-CE data instructions](https://github.com/jacobkrantz/VLN-CE#data)
for downloading and extracting the scenes. Place the files as follows:

```text
data/
├── datasets/
│   ├── R2R_VLNCE_v1-2_preprocessed_BERTidx/
│   │   ├── train/train_bertidx.json.gz
│   │   └── val_unseen/val_unseen_bertidx.json.gz
│   └── R2R_VLNCE_v1-2_preprocessed/
│       ├── train/train_gt.json.gz
│       └── val_unseen/val_unseen_gt.json.gz
├── scene_datasets/mp3d/<scene>/<scene>.glb
├── pretrained/ETP/mlm.sap_r2r/ckpts/model_step_82500.pt
├── wp_pred/check_cwp_bestdist_hfov90
├── ddppo-models/gibson-2plus-resnet50.pth
└── checkpoints/ckpt.iter12800.pth
```

The ETP, waypoint and depth-encoder weights are available in the ETPNav download.
CLIP downloads its ViT-B/32 weights on first use. Our navigation checkpoint is
distributed separately; see [Checkpoints](docs/CHECKPOINTS.md).

## Training

```bash
conda activate safe-vln
python scripts/run_experiment.py train \
  --data-dir ./data \
  --output-dir outputs/train
```

Training runs for **20,000 iterations**. Checkpoints from iteration **12,000 to
20,000** are saved every **200 iterations** in
`outputs/train/checkpoints/safe_vln_train/`; all 41 checkpoints are retained.

## Evaluation

Download [Safe-VLN-iter12800.zip](https://github.com/yueluhhxx/Safe-VLN/releases/download/checkpoint-iter12800/Safe-VLN-iter12800.zip),
extract it, and place `ckpt.iter12800.pth` in `data/checkpoints/`.
An [Aliyun Drive mirror](https://www.aliyundrive.com/t/HlRhz8xSxnX11rjTMVpb)
is also available until **2026-11-09 22:41 (UTC+8)**, with no extraction code.
Then run:

```bash
python scripts/run_experiment.py eval \
  --checkpoint data/checkpoints/ckpt.iter12800.pth \
  --data-dir ./data \
  --output-dir outputs/eval12800
```

This evaluates all **1839 val-unseen episodes** and checks the distributed
results for missing or duplicate episodes. Scores are saved in
`outputs/eval12800/evaluation_complete.json`.
See [evaluation settings](docs/EVALUATION.md) for the sensor and controller configuration.

To evaluate the full checkpoint window:

```bash
python scripts/run_experiment.py eval-window \
  --checkpoint-dir outputs/train/checkpoints/safe_vln_train \
  --data-dir ./data \
  --output-dir outputs/sweep
```

The default window is 12000–20000 with step 200. Each checkpoint gets a fresh
evaluation process and separate output directory. Results are written to
`outputs/sweep/ranking.csv` and `outputs/sweep/summary.json`, ranked by SR, then SPL.

Use `--gpus 0,1,2,3,4,5,6,7` to select the eight devices, `--port 29501` to
change the distributed port, or `--dry-run` to inspect the command without
running it. Output directories must be new or empty.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Acknowledgements

Thanks to [ETPNav](https://github.com/MarSaKi/ETPNav),
[Sim2Sim-VLNCE](https://github.com/jacobkrantz/Sim2Sim-VLNCE),
[VLN-CE](https://github.com/jacobkrantz/VLN-CE),
[Habitat](https://github.com/facebookresearch/habitat-lab), and
[DUET](https://github.com/cshizhe/VLN-DUET).
See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@article{yue2024safevln,
  title={Safe-VLN: Collision Avoidance for Vision-and-Language Navigation of Autonomous Robots Operating in Continuous Environments},
  author={Yue, Lu and Zhou, Dongliang and Xie, Liang and Zhang, Feitian and Yan, Ye and Yin, Erwei},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  doi={10.1109/LRA.2024.3387171}
}

@article{an2024etpnav,
  title={ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments},
  author={An, Dong and Wang, Hanqing and Wang, Wenguan and Wang, Zun and Huang, Yan and He, Keji and Wang, Liang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024}
}

@inproceedings{krantz2022sim2sim,
  title={Sim-2-Sim Transfer for Vision-and-Language Navigation in Continuous Environments},
  author={Krantz, Jacob and Lee, Stefan},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
```
