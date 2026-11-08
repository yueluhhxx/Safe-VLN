# Environment installation

This project uses a legacy Habitat stack on **Linux x86_64 with NVIDIA GPUs**.
The evaluated run used 8 NVIDIA A100 40 GB GPUs. macOS, Windows, newer Habitat
releases, and CPU-only navigation evaluation have not been validated.

The instructions below reconstruct the relevant dependencies from the evaluated
environment. A fresh environment built from these instructions has not yet
been validated against the reported SR.

## Recorded versions

| Component | Evaluated environment |
|---|---|
| Python | 3.8.0; the environment file permits compatible Python 3.8 patch releases |
| PyTorch | 2.0.1+cu117 |
| torchvision | 0.15.2+cu117 |
| PyTorch CUDA runtime | 11.7 |
| Habitat-Sim | 0.1.7, Python 3.8 headless, no Bullet |
| Habitat-Lab | `d6ed1c0a0e786f16f261de2beafe347f4186d0d8` |
| NumPy / SciPy | 1.23.5 / 1.10.1 |
| quaternion runtime | 2023.0.2 |
| Gym | 0.17.3 |
| transformers / pytorch-transformers | 4.42.4 / 1.2.0 |
| TensorFlow / TensorBoard | 2.13.0 / 2.13.0 |
| CLIP | Source matches `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` |

The original ETPNav README describes an older Python/PyTorch/Gym combination.
Use the recorded versions above for this release rather than mixing those two
recipes. NumPy must remain below 1.24 because the legacy evaluation code uses
`np.float`. The imported quaternion package reports `2023.0.2`; duplicated
metadata in the original shared environment made `pip list` report an older
version, so the runtime version is used here.

The original environment had mixed CUDA 11/12 Python packages and reported
cuDNN **8.9.2**. A clean official PyTorch 2.0.1 CUDA 11.7 installation commonly
uses cuDNN 8.5. This recipe deliberately avoids reproducing overlapping CUDA
package installations. Consequently, bitwise identity across rebuilt CUDA
environments is not guaranteed; record the environment check and validate a
complete checkpoint before comparing scores.

## 1. Create the environment

Run all commands from the root of this repository unless otherwise stated.
Install [Conda/Miniforge](https://github.com/conda-forge/miniforge#install) and
ensure an NVIDIA driver compatible with CUDA 11.7 is available. The driver is
a host prerequisite, not installed by these commands.

```bash
conda env create -f environment.yaml
conda activate safe-vln

python -m pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117

# fastdtw may need a local C/C++ compiler and Cython during installation.
python -m pip install Cython==0.29.36
python -m pip install -r requirements.txt
```

The YAML pins the exact Habitat-Sim headless build used in the run. If the
Conda solver cannot locate this old build, its upstream package is available
at the following direct URL; do not silently substitute a modern Habitat-Sim:

[Habitat-Sim 0.1.7 / Python 3.8 / headless / Linux package](https://conda.anaconda.org/aihabitat/linux-64/habitat-sim-0.1.7-py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2).

For distribution-specific rendering-library errors, follow the upstream
[Habitat-Sim installation documentation](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7#installation).
The server does not need a desktop display, but NVIDIA EGL/OpenGL libraries
must be available. Tk is included because a legacy module imports `turtle`,
even though the evaluation does not open a Tk window.

## 2. Install the pinned Habitat-Lab

```bash
mkdir -p third_party
git clone https://github.com/facebookresearch/habitat-lab.git third_party/habitat-lab
git -C third_party/habitat-lab checkout d6ed1c0a0e786f16f261de2beafe347f4186d0d8
git -C third_party/habitat-lab apply ../../patches/habitat-lab-optional-tensorboard.patch

cd third_party/habitat-lab
python setup.py develop --all --no-deps
cd ../..
```

The patch only removes the obsolete `tensorflow==1.13.1` and `tb-nightly`
installation requirements. TensorFlow and TensorBoard are pinned by this
repository instead. It does **not** change simulator actions, stopping criteria,
geodesic distances, rewards, or evaluation metrics.

The development machine's Habitat-Lab also contained unrelated extra-camera,
`Env.rebegin`, and `TeleportAction` customizations. The released code does not
reference those features, so they are not bundled. The project's own simulator
extension and training/evaluation controller remain in `habitat_extensions/`
and `vlnce_baselines/`.

## 3. Install CLIP

```bash
python -m pip install --no-deps \
  git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6
```

The installed `clip.py`, `model.py`, `simple_tokenizer.py`, and `__init__.py`
were compared byte-for-byte with this public commit. CLIP's ViT-B/32 weights
are downloaded on first use; allow access to the upstream download service or
pre-populate the normal CLIP cache on an offline machine.

## 4. Check the installation

```bash
python scripts/check_environment.py --skip-cuda
python scripts/check_environment.py --require-gpus 8
```

The first command checks imports and major pinned versions without initializing
a CUDA device. The second also runs a small CUDA tensor operation and records
the CUDA/cuDNN versions. Neither command downloads datasets or starts training.
A successful import check does not replace a complete navigation evaluation.

If a rebuild is needed, create a new environment; do not upgrade the original
environment in place. Do not install unrelated LLM packages, flash-attention,
xformers, or multiple CUDA package families into this research environment.

## Upstream references

- [ETPNav installation and data instructions](https://github.com/MarSaKi/ETPNav#setup)
- [Habitat-Lab pinned source and installation](https://github.com/facebookresearch/habitat-lab/tree/d6ed1c0a0e786f16f261de2beafe347f4186d0d8#installation)
- [Habitat-Sim v0.1.7 release](https://github.com/facebookresearch/habitat-sim/releases/tag/v0.1.7)
- [Official PyTorch previous-version installation commands](https://pytorch.org/get-started/previous-versions/#v201)
- [CLIP source](https://github.com/openai/CLIP)
- [Sim2Sim-VLNCE setup](https://github.com/jacobkrantz/Sim2Sim-VLNCE#setup)
