#!/usr/bin/env python3
"""Portable training and complete R2R-CE validation, without checkpoint deletion.

The published protocol is eight GPUs and two environments per rank. This
launcher intentionally does not expose silent batch-size or sensor changes.
Use --dry-run to inspect commands without importing PyTorch or creating files.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time


REPO = Path(__file__).resolve().parents[1]
DATASET_RELATIVE = Path('datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/val_unseen/val_unseen_bertidx.json.gz')
EXPECTED_EPISODES = 1839
WORLD_SIZE = 8


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, document):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint_index(path):
    """Mirror the released trainer's filename-based artifact index, not weights."""
    for component in Path(path).name.split('.'):
        if component.startswith('iter') and component[4:].isdigit():
            return max(0, int(component[4:]) // 2000 - 1)
    return 0


def visible_gpus(value):
    parts = value.split(',')
    if len(parts) != WORLD_SIZE or len(set(parts)) != WORLD_SIZE or any(not part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError('The reported protocol requires eight distinct GPU indices, e.g. 0,1,2,3,4,5,6,7')
    return ','.join(parts)


def config_files(mode):
    names = ['iter_train.yaml', 'iter_train_8gpu.yaml']
    if mode == 'train':
        names += ['perf_v3_selected.yaml', 'retrain_window_12000_20000.yaml']
    return [REPO / 'run_r2r' / name for name in names]


def build_command(mode, data, output, port, checkpoint=None, experiment=None):
    data, output = Path(data).resolve(), Path(output).resolve()
    experiment = experiment or ('safe_vln_train' if mode == 'train' else 'safe_vln_val_unseen')
    command = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1', '--node_rank=0',
               '--nproc_per_node=8', '--master_addr=127.0.0.1', '--master_port={}'.format(port),
               str(REPO / 'run.py'), '--exp_name', experiment, '--run-type', mode,
               '--exp-config', ','.join(str(path) for path in config_files(mode))]
    options = [
        ('SIMULATOR_GPU_IDS', '[0,1,2,3,4,5,6,7]'), ('TORCH_GPU_IDS', '[0,1,2,3,4,5,6,7]'),
        ('TORCH_GPU_ID', '0'), ('GPU_NUMBERS', '8'), ('NUM_ENVIRONMENTS', '2'),
        ('IL.batch_size', '2'), ('IL.lr', '1e-05'), ('IL.log_every', '200' if mode == 'train' else '2000'),
        ('TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING', 'True'),
        ('IL.tryout', 'True'), ('IL.reselection', 'True'), ('IL.reselection_max_retries', '1'),
        ('MODEL.SAFE_VLN.ENABLED', 'True'), ('MODEL.SAFE_VLN.DELTA', '1e-05'),
        ('MODEL.SAFE_VLN.SENSOR_HEIGHT', '0.24'),
        ('MODEL.pretrained_path', 'data/pretrained/ETP/mlm.sap_r2r/ckpts/model_step_82500.pt'),
        ('TRAINER_NAME', 'SS-ETP-Managed'), ('IL.iters', '20000'),
        ('IL.load_from_ckpt', 'False' if mode == 'train' else 'True'), ('IL.is_requeue', 'False'),
        ('IL.back_algo', 'teleport' if mode == 'train' else 'control'),
        ('CHECKPOINT_FOLDER', str(output / ('checkpoints' if mode == 'train' else 'unused_checkpoints')) + '/'),
        ('TENSORBOARD_DIR', str(output / 'tensorboard') + '/'),
        ('RESULTS_DIR', str(output / 'results') + '/'), ('VIDEO_DIR', str(output / 'video') + '/'),
    ]
    if mode == 'eval':
        if checkpoint is None:
            raise ValueError('Evaluation needs a checkpoint')
        path = str(Path(checkpoint).resolve())
        options += [
            ('EVAL.SPLIT', 'val_unseen'), ('EVAL.EPISODE_COUNT', '-1'), ('EVAL.fast_eval', 'False'),
            ('EVAL.SAVE_RESULTS', 'True'), ('EVAL.CKPT_PATH_DIR', path), ('EVAL_CKPT_PATH_DIR', path),
            ('EVAL.USE_CKPT_CONFIG', 'False'), ('TASK_CONFIG.SEED', '0'),
            ('TASK_CONFIG.DATASET.DATA_PATH', str(data / DATASET_RELATIVE)),
            ('IL.controller_motion_mode', 'etpnav_distance'), ('IL.ghost_route_mode', 'nearest_front'),
        ]
    command += [part for key, value in options for part in (key, value)]
    return command


def fixed_environment(gpus):
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=gpus, SAFE_VLN_BIND_NUMA='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
        NCCL_ASYNC_ERROR_HANDLING='1', TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
        PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
    for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        environment.pop(key, None)
    return environment


def required_inputs(data, mode):
    split = 'train' if mode == 'train' else 'val_unseen'
    names = [
        'wp_pred/check_cwp_bestdist_hfov90',
        'pretrained/ETP/mlm.sap_r2r/ckpts/model_step_82500.pt',
        'ddppo-models/gibson-2plus-resnet50.pth',
        'datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/{0}/{0}_bertidx.json.gz'.format(split),
        'datasets/R2R_VLNCE_v1-2_preprocessed/{0}/{0}_gt.json.gz'.format(split),
    ]
    return [Path(data) / name for name in names]


def preflight(data, mode, checkpoints, port):
    missing = [str(path) for path in required_inputs(data, mode) if not path.is_file()]
    if not (Path(data) / 'scene_datasets/mp3d').is_dir():
        missing.append(str(Path(data) / 'scene_datasets/mp3d'))
    missing += [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise ValueError('Missing inputs:\n' + '\n'.join(missing))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))


def new_output(path):
    path = Path(path)
    if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
        raise ValueError('Refusing to overwrite a nonempty output: ' + str(path))
    path.mkdir(parents=True, exist_ok=True)


def create_work(output, data):
    """Keep hard-coded relative data paths portable without altering input data."""
    work = output / 'work'
    work.mkdir()
    for name in ('run.py', 'run_r2r', 'vlnce_baselines', 'habitat_extensions', 'bert_config'):
        target = REPO / name
        (work / name).symlink_to(target, target_is_directory=target.is_dir())
    (work / 'data/logs/running_log').mkdir(parents=True)
    for source in data.iterdir():
        if source.name != 'logs':
            target = source.resolve()
            (work / 'data' / source.name).symlink_to(target, target_is_directory=target.is_dir())
    return work


def terminate_child(child):
    """Only terminate the new session created by this invocation."""
    if child.pid <= 1 or child.pid == os.getpgrp():
        raise RuntimeError('Unsafe child process-group identity')
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            continue
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            return


def run_one(mode, data, output, port, gpus, checkpoint=None, experiment=None):
    data, output = data.resolve(), output.resolve()
    new_output(output)
    work = create_work(output, data)
    command = build_command(mode, data, output, port, checkpoint, experiment)
    experiment = command[command.index('--exp_name') + 1]
    manifest = dict(mode=mode, created_at=utc_now(), command=command, working_directory=str(work),
        data_dir=str(data), visible_gpus=gpus, world_size=8, environments_per_gpu=2,
        seed=0, checkpoint_deletion=False,
        config_sha256={str(path.relative_to(REPO)): sha256_file(path) for path in config_files(mode)},
        input_sha256={str(path.relative_to(data)): sha256_file(path) for path in required_inputs(data, mode)})
    if checkpoint is not None:
        manifest.update(checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
                        checkpoint_index=checkpoint_index(checkpoint))
    atomic_json(output / 'run_manifest.json', manifest)
    atomic_json(output / 'status.json', dict(state='running', started_at=utc_now(), mode=mode))
    print('Launching {}. Log: {}'.format(mode, output / (mode + '.log')), flush=True)
    started = time.monotonic()
    child = None
    try:
        with (output / (mode + '.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=str(work), env=fixed_environment(gpus),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            exit_code = child.wait()
        if exit_code:
            raise RuntimeError('Evaluator/trainer exited {}. Inspect {}.'.format(exit_code, output / (mode + '.log')))
        for path, digest in manifest['config_sha256'].items():
            if sha256_file(REPO / path) != digest:
                raise RuntimeError('Configuration changed while running: ' + path)
        for path, digest in manifest['input_sha256'].items():
            if sha256_file(data / path) != digest:
                raise RuntimeError('Input changed while running: ' + path)
        complete = dict(state='complete', completed_at=utc_now(), elapsed_seconds=time.monotonic()-started,
                        exit_code=0, manifest=manifest)
        if mode == 'eval':
            if sha256_file(checkpoint) != manifest['checkpoint_sha256']:
                raise RuntimeError('Checkpoint changed during evaluation')
            sys.path.insert(0, str(REPO))
            from tools.validate_full_eval import validate_full_evaluation
            validation = validate_full_evaluation(output / 'results' / experiment, data / DATASET_RELATIVE,
                checkpoint_index=manifest['checkpoint_index'], world_size=8, expected_episode_count=EXPECTED_EPISODES)
            complete['validation'] = validation
            atomic_json(output / 'evaluation_complete.json', complete)
            metrics = validation['recomputed_metrics']
            print('Full val-unseen: SR={:.4f}% SPL={:.4f}% nDTW={:.4f}%'.format(
                metrics['success']*100, metrics['spl']*100, metrics['ndtw']*100), flush=True)
        else:
            folder = output / 'checkpoints' / experiment
            expected = [folder / 'ckpt.iter{}.pth'.format(i) for i in range(12000, 20001, 200)]
            if any(not path.is_file() or path.stat().st_size == 0 for path in expected):
                raise RuntimeError('Training finished without all 41 requested checkpoints')
            complete['checkpoints'] = [str(path) for path in expected]
            complete['checkpoint_count'] = len(expected)
            complete['retention'] = 'keep_all'
            atomic_json(output / 'training_complete.json', complete)
        atomic_json(output / 'status.json', dict(state='complete', completed_at=utc_now(), mode=mode))
        return complete
    except BaseException as error:
        if child is not None:
            terminate_child(child)
        atomic_json(output / 'status.json', dict(state='failed', updated_at=utc_now(), error=repr(error)))
        raise


def window_candidates(directory, start, end, step):
    if start <= 0 or step <= 0 or end < start or (end-start) % step:
        raise ValueError('Window requires positive start/step and an inclusive, divisible range')
    return [(iteration, directory / 'ckpt.iter{}.pth'.format(iteration)) for iteration in range(start, end+1, step)]


def write_ranking(output, rows, expected):
    ranked = sorted(rows, key=lambda row: (row['success_count'], row['metrics']['spl'], row['iteration']), reverse=True)
    document = dict(state='complete' if len(rows)==expected else 'running', completed=len(rows),
        expected_count=expected, all_candidates_evaluated=len(rows)==expected,
        selection_rule=['SR descending', 'SPL descending', 'iteration descending'],
        selection_note='Checkpoint selection on val-unseen; this is not an independent test-set estimate.',
        retention='keep_all', best=ranked[0] if ranked else None, ranked_results=ranked)
    atomic_json(output / 'summary.json', document)
    with (output / 'ranking.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['rank', 'iteration', 'episodes', 'successes', 'SR_percent', 'SPL_percent', 'nDTW_percent', 'checkpoint'])
        for rank, row in enumerate(ranked, 1):
            metrics = row['metrics']
            writer.writerow([rank, row['iteration'], EXPECTED_EPISODES, row['success_count'],
                metrics['success']*100, metrics['spl']*100, metrics['ndtw']*100, row['checkpoint']])
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'eval', 'eval-window'))
    parser.add_argument('--data-dir', type=Path, default=REPO / 'data')
    parser.add_argument('--output-dir', type=Path, help='Must be new or empty; defaults to outputs/<mode>.')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--gpus', type=visible_gpus, default='0,1,2,3,4,5,6,7')
    parser.add_argument('--port', type=int, default=29500)
    parser.add_argument('--start', type=int, default=12000)
    parser.add_argument('--end', type=int, default=20000)
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error('--port must be between 1024 and 65535')
    if args.mode == 'eval' and args.checkpoint is None:
        parser.error('eval requires --checkpoint')
    if args.mode == 'eval-window' and args.checkpoint_dir is None:
        parser.error('eval-window requires --checkpoint-dir')
    data = args.data_dir.expanduser().resolve()
    output = (args.output_dir or REPO / 'outputs' / args.mode).expanduser().resolve()
    if args.mode == 'eval-window':
        candidates = window_candidates(args.checkpoint_dir.expanduser().resolve(), args.start, args.end, args.step)
        planned = [(i, path, output / 'iter_{}'.format(i), 'iter{}_val_unseen'.format(i)) for i, path in candidates]
    else:
        path = args.checkpoint.expanduser().resolve() if args.checkpoint else None
        planned = [(None, path, output, None)]
    mode = 'train' if args.mode == 'train' else 'eval'
    if args.dry_run:
        print(json.dumps(dict(mode=args.mode, visible_gpus=args.gpus, keep_all_checkpoints=True,
            commands=[build_command(mode, data, destination, args.port, path, name)
                      for _, path, destination, name in planned]), indent=2))
        return
    preflight(data, mode, [path for _, path, _, _ in planned if path is not None], args.port)
    def interrupted(signum, frame):
        raise RuntimeError('Launcher interrupted by signal {}'.format(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if args.mode != 'eval-window':
        _, path, destination, name = planned[0]
        run_one(mode, data, destination, args.port, args.gpus, path, name)
        return
    new_output(output)
    rows = []
    write_ranking(output, rows, len(planned))
    try:
        for iteration, path, destination, name in planned:
            complete = run_one('eval', data, destination, args.port, args.gpus, path, name)
            metrics = complete['validation']['recomputed_metrics']
            rows.append(dict(iteration=iteration, checkpoint=str(path),
                checkpoint_sha256=complete['manifest']['checkpoint_sha256'], episode_count=EXPECTED_EPISODES,
                success_count=round(metrics['success']*EXPECTED_EPISODES), metrics=metrics,
                certificate=str(destination / 'evaluation_complete.json')))
            summary = write_ranking(output, rows, len(planned))
        atomic_json(output / 'evaluation_complete.json', summary)
        print('Best checkpoint: {}'.format(summary['best']['checkpoint']), flush=True)
    except BaseException as error:
        summary = write_ranking(output, rows, len(planned))
        summary.update(state='failed', error=repr(error))
        atomic_json(output / 'summary.json', summary)
        raise


if __name__ == '__main__':
    main()
