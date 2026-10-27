"""V3-only paths, task configuration, process environments and artifact validation."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime,timezone,timedelta

RUN=Path(__file__).resolve().parents[1]
ROOT=RUN.parent
V2=ROOT/'ACT_v2_results'
PREFIX=Path('/HOME/sysu_qling/sysu_qling_3/HDD_POOL/anaconda3/envs/act-repro')
PYTHON=str(PREFIX/'bin/python')
PLAN=json.loads((RUN/'experiment_plan.json').read_text())
ALL_TASKS=PLAN['tasks']
TASKS=[s for s in ALL_TASKS if s.get('enabled',True)]


def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.%d.tmp'%os.getpid())
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))
    os.replace(str(temp),str(path))


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def task_spec(name):return next(s for s in ALL_TASKS if s['name']==name or s['task_name']==name)


def worker_environment():
    env=dict(os.environ)
    for key in list(env):
        if key.startswith('MUJOCO_EGL') or key.startswith('__EGL') or key in [
            'PYTHONPATH','EGL_PLATFORM','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','NVIDIA_TF32_OVERRIDE','LD_PRELOAD']:
            env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',
        MPLBACKEND='Agg',MPLCONFIGDIR=str(RUN/'runtime/matplotlib'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',LP_NUM_THREADS='2',GALLIUM_DRIVER='llvmpipe',
        LD_LIBRARY_PATH=str(RUN/'runtime/mesa/usr/lib/x86_64-linux-gnu')+':'+str(PREFIX/'lib/act-graphics'))
    return env


def activate_renderer():
    import ctypes
    library=RUN/'runtime/mesa/usr/lib/x86_64-linux-gnu'
    ctypes.CDLL(str(library/'libglapi.so.0'),mode=ctypes.RTLD_GLOBAL)
    original=ctypes.cdll._dlltype
    def route(name,*args,**kwargs):
        if isinstance(name,str) and name.startswith('libOSMesa.so'):name=str(library/'libOSMesa.so.8')
        return original(name,*args,**kwargs)
    ctypes.cdll._dlltype=route


def official_args(directory,spec,epochs=2000,evaluation=False):
    a=dict(PLAN['training'])
    for key in ['backbone','camera_names']:a.pop(key)
    a.update(eval=evaluation,onscreen_render=False,ckpt_dir=str(directory),task_name=spec['task_name'],
             num_epochs=epochs,temporal_agg=bool(evaluation))
    return a


def install_official_argv(args):
    import sys
    sys.argv=[str(ROOT/'imitate_episodes.py')]
    for key,value in args.items():
        if isinstance(value,bool):
            if value:sys.argv.append('--'+key)
        else:sys.argv+=['--'+key,str(value)]


def check_sources():
    for name,digest in json.loads((RUN/'provenance/source_manifest.json').read_text()).items():
        assert sha256(ROOT/name)==digest,'Official source changed: '+name


def validate_model(directory,epochs=2000):
    import pickle,math
    import numpy as np
    import torch
    directory=Path(directory);config=json.loads((directory/'training_config.json').read_text())
    spec=task_spec(config['task_name'])
    rows=[json.loads(s) for s in (directory/'epoch_metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows]==list(range(epochs)) and config['num_epochs']==epochs
    assert all(math.isfinite(v) for r in rows for key in ['train','validation'] for v in r[key].values())
    assert config['initialization']['prior_experiment_checkpoint_loaded'] is False
    for filename in ['policy_best.ckpt','policy_last.ckpt']:
        state=torch.load(str(directory/filename),map_location='cpu')
        assert state and all(torch.isfinite(v).all().item() for v in state.values())
        assert list(state['model.query_embed.weight'].shape)==[100,512]
        del state
    with (directory/'dataset_stats.pkl').open('rb') as f:stats=pickle.load(f)
    with (RUN/'reports/data'/spec['name']/'reference_stats.pkl').open('rb') as f:expected=pickle.load(f)
    for key in ['qpos_mean','qpos_std','action_mean','action_std']:
        assert stats[key].shape==(14,) and np.isfinite(stats[key]).all()
        assert np.array_equal(stats[key],expected[key]),'Normalization from wrong dataset: '+spec['task_name']
        if key.endswith('_std'):assert (stats[key]>0).all()
    best=min(rows,key=lambda r:r['validation']['loss'])
    return dict(epochs=epochs,task_name=spec['task_name'],best_epoch=best['epoch'],best_validation_loss=best['validation']['loss'],
        checkpoint_sha256=sha256(directory/'policy_best.ckpt'),stats_sha256=sha256(directory/'dataset_stats.pkl'))
