"""V2-only paths, atomic state, immutable source checks and task definitions."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parent
PREFIX = Path('/HOME/sysu_qling/sysu_qling_3/HDD_POOL/anaconda3/envs/act-repro')
PYTHON = str(PREFIX / 'bin/python')
TRAINING_SEEDS = json.loads((RUN/'experiment_plan.json').read_text())['training']['training_seeds']
assert TRAINING_SEEDS and len(set(TRAINING_SEEDS))==len(TRAINING_SEEDS)
NUM_SEEDS = len(TRAINING_SEEDS)
MODELS = [{'name': 'k%d_seed%d' % (k, s), 'chunk': k, 'seed': s}
          for k in [100, 1, 400] for s in TRAINING_SEEDS]
EVALS = ([dict(m, name=m['name']+'_temporal', model=m['name'], temporal=True, group='main')
          for m in MODELS if m['chunk'] == 100] +
         [dict(m, name=m['name']+'_no_temporal', model=m['name'], temporal=False, group='ablation')
          for m in MODELS])


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def write_json(path, obj):
    path = Path(path)
    temp = path.with_name(path.name+'.%d.tmp' % os.getpid())
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(str(temp), str(path))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8*1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def worker_environment(renderer=None, threads=1):
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    for key in ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'NVIDIA_TF32_OVERRIDE']:
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), NUMEXPR_NUM_THREADS=str(threads),
               MPLBACKEND='Agg', MPLCONFIGDIR=str(RUN/'runtime/matplotlib'),
               MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa',
               LD_LIBRARY_PATH=str(PREFIX/'lib/act-graphics'))
    selection = RUN/'reports/render_selection.json'
    if renderer is None and selection.exists():
        renderer = json.loads(selection.read_text())['renderer']
    if renderer == 'llvmpipe':
        env['LD_LIBRARY_PATH'] = str(RUN/'runtime/mesa/usr/lib/x86_64-linux-gnu')+':'+env['LD_LIBRARY_PATH']
        env.update(GALLIUM_DRIVER='llvmpipe', LP_NUM_THREADS='2')
    else:
        env.pop('GALLIUM_DRIVER', None)
        env.pop('LP_NUM_THREADS', None)
    if renderer == 'egl':
        env.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID='0')
    env['ACT_V2_RENDERER'] = renderer or 'softpipe'
    return env


def activate_renderer():
    """Process-only OSMesa library routing; never edit Conda or system libraries.

    Conda Python's DT_RPATH can beat LD_LIBRARY_PATH. Route only PyOpenGL's
    OSMesa library request to the audited sidecar; preload its matching GLAPI.
    """
    if os.environ.get('ACT_V2_RENDERER') != 'llvmpipe':
        return
    import ctypes
    directory = RUN/'runtime/mesa/usr/lib/x86_64-linux-gnu'
    ctypes.CDLL(str(directory/'libglapi.so.0'), mode=ctypes.RTLD_GLOBAL)
    original = ctypes.cdll._dlltype
    def route(name, *args, **kwargs):
        if isinstance(name, str) and name.startswith('libOSMesa.so'):
            name = str(directory/'libOSMesa.so.8')
        return original(name, *args, **kwargs)
    ctypes.cdll._dlltype = route


def official_args(directory, chunk, seed, epochs=2000, temporal=False, evaluation=False):
    return dict(eval=evaluation, onscreen_render=False, ckpt_dir=str(directory),
                policy_class='ACT', task_name='sim_transfer_cube_scripted', batch_size=8,
                seed=seed, num_epochs=epochs, lr=1e-5, kl_weight=10, chunk_size=chunk,
                hidden_dim=512, dim_feedforward=3200, temporal_agg=temporal)


def install_official_argv(args):
    import sys
    sys.argv = [str(ROOT/'imitate_episodes.py')]
    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                sys.argv.append('--'+key)
        else:
            sys.argv += ['--'+key, str(value)]


def check_sources():
    manifest = json.loads((RUN/'provenance/source_manifest.json').read_text())
    for name, digest in manifest.items():
        assert sha256(ROOT/name) == digest, 'Experiment source changed: '+name


def validate_model(directory, epochs=2000):
    import math
    import pickle
    import numpy as np
    import torch
    directory = Path(directory)
    config = json.loads((directory/'training_config.json').read_text())
    metrics = [json.loads(line) for line in (directory/'epoch_metrics.jsonl').read_text().splitlines()]
    assert [m['epoch'] for m in metrics] == list(range(epochs))
    assert all(math.isfinite(v) for m in metrics for series in ['train', 'validation'] for v in m[series].values())
    assert config['initialization']['e1_checkpoint_loaded'] is False
    assert config['num_epochs'] == epochs
    for name in ['policy_best.ckpt', 'policy_last.ckpt']:
        state = torch.load(str(directory/name), map_location='cpu')
        assert state and all(torch.isfinite(v).all().item() for v in state.values())
        assert list(state['model.query_embed.weight'].shape) == [config['chunk_size'], 512]
        del state
    with (directory/'dataset_stats.pkl').open('rb') as f:
        stats = pickle.load(f)
    for key in ['action_mean', 'action_std', 'qpos_mean', 'qpos_std']:
        assert stats[key].shape == (14,) and np.isfinite(stats[key]).all()
        if key.endswith('_std'):
            assert (stats[key] > 0).all()
    best = min(metrics, key=lambda m: m['validation']['loss'])
    return dict(epochs=epochs, best_epoch=best['epoch'], best_validation_loss=best['validation']['loss'],
                checkpoint_sha256=sha256(directory/'policy_best.ckpt'),
                stats_sha256=sha256(directory/'dataset_stats.pkl'))
