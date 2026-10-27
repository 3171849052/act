"""Per-task, low-priority read-only data validation; never edits existing HDF5."""
import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path
from common import ROOT,RUN,TASKS,task_spec,now,write_json,sha256,check_sources
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',required=True,choices=[s['name'] for s in TASKS]);a=parser.parse_args()
    os.nice(10)
    import h5py
    import numpy as np
    import torch
    import utils
    from constants import SIM_TASK_CONFIGS
    torch.set_num_threads(1)
    spec=task_spec(a.task);task=SIM_TASK_CONFIGS[spec['task_name']]
    assert task['dataset_dir']==spec['dataset_dir'] and task['episode_len']==spec['episode_len']
    assert task['num_episodes']==50 and task['camera_names']==['top']
    directory=Path(task['dataset_dir']);output=RUN/'reports/data'/spec['name'];output.mkdir(parents=True,exist_ok=True)
    if not directory.exists():
        write_json(output/'validation.json',dict(passed=False,blocked=True,reason='Dataset directory missing',task=spec,checked_at=now()))
        raise RuntimeError('Dataset missing: '+str(directory))
    expected={'episode_%d.hdf5'%i for i in range(50)}
    assert {p.name for p in directory.glob('episode_*.hdf5')}==expected,'Need 50 complete named HDF5 files'
    assert shutil.disk_usage(str(RUN)).free>50*1024**3
    check_sources()
    report=dict(passed=False,task=spec,started_at=now(),files=[],split_seed=1,
                sim_flag_meaning='sim=True identifies simulation, not scripted versus human provenance')
    length=task['episode_len']
    for i in range(50):
        path=directory/('episode_%d.hdf5'%i)
        row=dict(id=i,path=str(path),bytes=path.stat().st_size)
        with h5py.File(str(path),'r') as f:
            assert bool(f.attrs['sim'])
            for name in ['action','observations/qpos','observations/qvel']:
                array=f[name][:]
                assert array.shape==(length,14) and np.isfinite(array).all()
                assert np.max(np.abs(array))<1e6,'Numerically implausible trajectory values'
                row[name]=dict(shape=list(array.shape),dtype=str(array.dtype),max_abs=float(np.max(np.abs(array))))
            image=f['observations/images/top'];assert image.shape==(length,480,640,3) and image.dtype==np.uint8
            for start in range(0,length,8):
                block=image[start:start+8]
                assert block.shape[0]>0 and np.ptp(block)>0
            row['image_shape']=list(image.shape);row['image_dtype']=str(image.dtype)
        row['sha256']=sha256(path);report['files'].append(row)
        print('Validated',spec['task_name'],i,flush=True)
    utils.set_seed(1)
    train,val,stats,is_sim=utils.load_data(str(directory),50,['top'],8,8)
    assert is_sim
    split=dict(train=train.dataset.episode_ids.tolist(),validation=val.dataset.episode_ids.tolist())
    assert len(split['train'])==40 and len(split['validation'])==10
    assert sorted(split['train']+split['validation'])==list(range(50))
    report['split']=split;report['batch_shapes']={}
    for name,loader in [('train',train),('validation',val)]:
        batch=next(iter(loader));shapes=[list(x.shape) for x in batch]
        assert shapes==[[8,1,3,480,640],[8,14],[8,length,14],[8,length]]
        assert all(torch.isfinite(x).all() for x in batch)
        report['batch_shapes'][name]=shapes
    with (output/'reference_stats.pkl').open('wb') as f:pickle.dump(stats,f)
    report.update(passed=True,completed_at=now(),normalization_scope='Official all-50 trajectory statistics, own dataset only')
    write_json(output/'validation.json',report)
    print('DATA VALIDATION PASSED',spec['name'],flush=True)


if __name__=='__main__':main()
