import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    import torch
    import h5py
    import numpy as np
    import mujoco
    import dm_control
    from constants import DATA_DIR, SIM_TASK_CONFIGS
    from utils import load_data, set_seed, sample_box_pose
    from sim_env import BOX_POSE, make_sim_env
    import matplotlib.pyplot as plt
    report = {'project_root': str(ROOT), 'python': sys.executable,
              'python_version': sys.version, 'platform': platform.platform(),
              'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
              'DATA_DIR': DATA_DIR, 'versions': {}, 'time': time.time(),
              'render_backend': os.environ.get('MUJOCO_GL')}
    for pkg in ['torch','torchvision','mujoco','dm-control','h5py','numpy','matplotlib']:
        report['versions'][pkg] = importlib.metadata.version(pkg)
    for label, command in [('nvidia_smi',['nvidia-smi']), ('disk',['df','-h',str(ROOT)]),
                           ('git_status',['git','status','--short']),('cpu',['lscpu'])]:
        output = subprocess.check_output(command, cwd=ROOT, text=True)
        (RUN/'reports'/('preflight_'+label+'.txt')).write_text(output)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    report['gpu'] = torch.cuda.get_device_name(0)
    report['torch_cuda'] = torch.version.cuda
    x = torch.randn(64,64,device='cuda'); assert torch.isfinite(x @ x).all()
    torch.cuda.synchronize()
    task = SIM_TASK_CONFIGS['sim_transfer_cube_scripted']
    data = Path(task['dataset_dir'])
    assert data.resolve() == (ROOT/'data/sim_transfer_cube_scripted').resolve()
    assert {p.name for p in data.glob('episode_*.hdf5')} == {f'episode_{i}.hdf5' for i in range(50)}
    episodes = []
    for i in range(50):
        p = data/f'episode_{i}.hdf5'
        with h5py.File(p,'r') as f:
            assert bool(f.attrs['sim'])
            for key in ['action','observations/qpos','observations/qvel']:
                a = f[key][()]
                assert a.shape == (400,14) and np.isfinite(a).all(), (p,key,a.shape)
            im = f['observations/images/top']
            assert im.shape == (400,480,640,3) and im.dtype == np.uint8
            # Read every compressed image chunk, rather than trusting metadata.
            for start in range(0,400,20):
                frames = im[start:start+20]
                assert frames.shape[0] == 20 and frames.max() > frames.min()
        digest = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024), b''): digest.update(block)
        episodes.append({'id':i,'bytes':p.stat().st_size,'sha256':digest.hexdigest()})
        print('Validated full HDF5',i,flush=True)
    set_seed(1)  # Official main() uses seed 1 for the split, then seed 0 for training.
    train,val,stats,is_sim = load_data(str(data),50,['top'],8,8)
    report['split'] = {'train':train.dataset.episode_ids.tolist(),'validation':val.dataset.episode_ids.tolist(),
                       'split_seed':1,'training_seed':0}
    report['batch_shapes'] = {}
    for name,loader in [('train',train),('validation',val)]:
        batch = next(iter(loader))
        report['batch_shapes'][name] = [list(t.shape) for t in batch]
        assert all(torch.isfinite(t).all() for t in batch)
    assert is_sim
    BOX_POSE[0] = sample_box_pose()
    env = make_sim_env('sim_transfer_cube_scripted')
    ts = env.reset()
    top = ts.observation['images']['top']
    assert top.shape == (480,640,3) and top.dtype == np.uint8 and top.max()>top.min()
    plt.imsave(RUN/'reports/preflight_sim_top.png', top)
    ts = env.step(ts.observation['qpos'])
    assert np.isfinite(ts.observation['qpos']).all()
    report['environment'] = {'reset':True,'step':True,'top_shape':list(top.shape),'max_reward':env.task.max_reward}
    env.close()
    report['free_disk_bytes'] = shutil.disk_usage(ROOT).free
    assert report['free_disk_bytes'] > 30*1024**3
    report['episodes'] = episodes
    report['passed'] = True
    (RUN/'reports/preflight.json').write_text(json.dumps(report,indent=2))
    print('PREFLIGHT PASSED',flush=True)

if __name__ == '__main__':
    main()
