#!/usr/bin/env python3
"""Read-only import/version checks; no datasets, model downloads, or training."""
import argparse
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys


EXPECTED = {
    'numpy': '1.23.5', 'scipy': '1.10.1', 'gym': '0.17.3',
    'habitat': '0.1.7', 'habitat-sim': '0.1.7',
    'torch': '2.0.1', 'torchvision': '0.15.2',
    'transformers': '4.42.4', 'pytorch-transformers': '1.2.0',
    'tensorboard': '2.13.0', 'tensorflow': '2.13.0',
}
IMPORTS = (
    'numpy', 'scipy', 'quaternion', 'gym', 'cv2', 'tkinter',
    'habitat', 'habitat_sim', 'habitat_baselines', 'torch', 'torchvision',
    'clip', 'transformers', 'pytorch_transformers', 'tensorflow',
    'fastdtw', 'dtw', 'jsonlines', 'lmdb', 'msgpack_numpy', 'networkx',
)
CLIP_HASHES = {
    'clip.py': '9540f200fbf8145479fa655382a56dab048d238cc698b9cbd8df3b6d86d3f1b6',
    'model.py': '9902cbe5ee90a1da2aa3e6f043e8a23dc1f8831193b963785c9af03d5c7bef2c',
    'simple_tokenizer.py': 'd1b09f10ed3d1e2c343619cb98cbfacb92363dd8607526a43aa7fe2d60d15bad',
    '__init__.py': 'd7930eb290ebed432116795c91473200d098451a94c0aedd4ae0f934f97955e8',
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-cuda', action='store_true')
    parser.add_argument('--require-gpus', type=int, default=1)
    args = parser.parse_args(argv)
    if args.require_gpus < 1:
        parser.error('--require-gpus must be positive')
    # This must happen before importing either PyTorch or TensorFlow.
    if args.skip_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    errors, warnings = [], []
    versions = {}
    if sys.version_info[:2] != (3, 8):
        errors.append('Use Python 3.8 for the pinned Habitat-Sim binary.')
    for name, expected in EXPECTED.items():
        try:
            version = metadata.version(name)
            versions[name] = version
            if version.split('+')[0] != expected:
                errors.append('{}: expected {}, found {}'.format(name, expected, version))
        except metadata.PackageNotFoundError:
            errors.append('Missing distribution: ' + name)
    imported = {}
    for name in IMPORTS:
        try:
            imported[name] = importlib.import_module(name)
        except Exception as error:
            errors.append('{} import failed: {}'.format(name, error))
    quaternion = imported.get('quaternion')
    if quaternion is not None:
        versions['quaternion_runtime'] = quaternion.__version__
        if quaternion.__version__ != '2023.0.2':
            errors.append('quaternion runtime differs from 2023.0.2')
    clip = imported.get('clip')
    if clip is not None:
        for name, expected in CLIP_HASHES.items():
            path = Path(clip.__file__).parent / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append('CLIP source differs from pinned commit: ' + name)
    try:
        importlib.import_module('habitat_extensions')
        importlib.import_module('vlnce_baselines')
    except Exception as error:
        errors.append('Project import failed: {}'.format(error))
    cuda = {'checked': not args.skip_cuda}
    torch = imported.get('torch')
    if torch is not None and not args.skip_cuda:
        cuda.update(available=torch.cuda.is_available(), device_count=torch.cuda.device_count(),
                    torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
        if cuda['device_count'] < args.require_gpus:
            errors.append('Requested {} GPU(s), found {}'.format(args.require_gpus, cuda['device_count']))
        if cuda['available']:
            try:
                value = torch.ones(2, device='cuda').sum().item()
                if value != 2.0:
                    errors.append('CUDA tensor smoke test produced an incorrect value')
            except Exception as error:
                errors.append('CUDA tensor operation failed: {}'.format(error))
        if torch.version.cuda != '11.7':
            warnings.append('CUDA runtime differs from the evaluated 11.7 build.')
        if cuda['cudnn'] != 8902:
            warnings.append('cuDNN differs from the original mixed environment (8902); see docs/INSTALL.md.')
    report = {'status': 'passed' if not errors else 'failed', 'python': sys.version.split()[0],
              'versions': versions, 'cuda': cuda, 'warnings': warnings, 'errors': errors,
              'scope': 'Imports and runtime checks only; not a fresh-install or navigation-score certificate.'}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
