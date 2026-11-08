#!/usr/bin/env python3
"""Export a trusted training checkpoint without optimizer or machine config.

Tensor names, values and dtypes are preserved. The original file is read-only.
Only use this tool with a trusted checkpoint: its input uses Python pickle.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--expected-sha256', required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    metadata = output.with_suffix('.json')
    checksum = output.with_suffix('.sha256')
    if source == output or any(p.exists() for p in (output, metadata, checksum)):
        parser.error('Output and sidecar files must not exist; original is never overwritten.')
    before = source.stat()
    source_hash = sha256(source)
    if source_hash != args.expected_sha256:
        parser.error('Source SHA256 does not match the expected trusted checkpoint.')
    import torch
    original = torch.load(source, map_location='cpu', weights_only=False)
    state = original['state_dict']
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError('Expected a nonempty tensor-only model state dictionary.')
    public = {'state_dict': {key: value.detach().cpu() for key, value in state.items()},
              'iteration': int(original['iteration']), 'checkpoint_format_version': 1}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        torch.save(public, stream)
    restored = torch.load(output, map_location='cpu', weights_only=True)
    if restored['iteration'] != original['iteration'] or restored['state_dict'].keys() != state.keys():
        raise RuntimeError('Exported metadata or tensor keys changed.')
    for key, value in state.items():
        exported = restored['state_dict'][key]
        if exported.dtype != value.dtype or not torch.equal(exported, value.cpu()):
            raise RuntimeError('Tensor changed during export: ' + key)
    after = source.stat()
    if any(getattr(before, key) != getattr(after, key) for key in ('st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')):
        raise RuntimeError('Source checkpoint changed during export.')
    digest = sha256(output)
    record = {'filename': output.name, 'iteration': public['iteration'],
              'format': 'PyTorch tensor state_dict + iteration; evaluation/fine-tuning only',
              'sha256': digest, 'size_bytes': output.stat().st_size,
              'source_checkpoint_sha256': source_hash, 'tensor_count': len(state),
              'tensor_values_and_dtypes_verified_equal': True,
              'optimizer_included': False, 'machine_configuration_included': False}
    with metadata.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    with checksum.open('x') as stream:
        stream.write('{}  {}\n'.format(digest, output.name))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
