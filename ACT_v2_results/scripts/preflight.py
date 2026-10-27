"""Read-only dataset, provenance, loader, CUDA, and historical-result audit."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import importlib.metadata as metadata
from common import ROOT, RUN, now, write_json, sha256
sys.path.insert(0, str(ROOT))


def main():
    import h5py
    import numpy as np
    import torch
    import utils
    from constants import SIM_TASK_CONFIGS
    result = dict(passed=False, started_at=now(), cwd=os.getcwd(), project=str(ROOT),
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip(),
                  python=sys.executable, versions={}, dataset=[], split_seed=1)
    for name in ['torch', 'torchvision', 'numpy', 'h5py', 'mujoco', 'dm-control', 'PyOpenGL', 'glfw']:
        result['versions'][name] = metadata.version(name)
    result['torch_cuda_version'] = torch.version.cuda
    assert torch.cuda.is_available(), 'CUDA unavailable for ACT training'
    result['gpu'] = dict(name=torch.cuda.get_device_name(0),
                         total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    result['cpu_quota'] = Path('/sys/fs/cgroup/cpu.max').read_text().strip()
    result['memory_limit'] = Path('/sys/fs/cgroup/memory.max').read_text().strip()
    disk = shutil.disk_usage(str(RUN))
    result['disk'] = dict(total=disk.total, used=disk.used, free=disk.free,
                          conservative_required_bytes=120*1024**3,
                          note='Shared filesystem free space; user/project quota may be separate')
    assert disk.free > 120*1024**3
    comparison = json.loads((RUN/'provenance/official_comparison.json').read_text())
    assert all(v['identical'] for k, v in comparison['files'].items() if k != 'constants.py')
    result['official_upstream_commit'] = comparison['commit']
    # Snapshot all Python model/training sources and assets for reproducibility.
    sources = sorted(ROOT.glob('*.py')) + sorted((ROOT/'detr').rglob('*.py'))
    write_json(RUN/'provenance/source_manifest.json', {str(p.relative_to(ROOT)): sha256(p) for p in sources})
    for p in sources:
        target = RUN/'provenance/local_sources'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(p), str(target))
    write_json(RUN/'provenance/assets_manifest.json', {str(p.relative_to(ROOT)): sha256(p)
        for p in sorted((ROOT/'assets').rglob('*')) if p.is_file()})
    task = SIM_TASK_CONFIGS['sim_transfer_cube_scripted']
    assert task['num_episodes'] == 50 and task['episode_len'] == 400 and task['camera_names'] == ['top']
    data = Path(task['dataset_dir'])
    assert {p.name for p in data.glob('episode_*.hdf5')} == {'episode_%d.hdf5'%i for i in range(50)}
    old = json.loads((ROOT/'E1_results/reports/preflight.json').read_text())
    old_hashes = {item['id']: item['sha256'] for item in old['episodes']}
    for i in range(50):
        path = data/('episode_%d.hdf5'%i)
        digest = sha256(path)
        assert digest == old_hashes[i], 'Dataset differs from E1 snapshot: '+str(i)
        with h5py.File(str(path), 'r') as f:
            assert bool(f.attrs['sim'])
            for key in ['action', 'observations/qpos', 'observations/qvel']:
                assert f[key].shape == (400, 14) and np.isfinite(f[key][:]).all()
            images = f['observations/images/top']
            assert images.shape == (400, 480, 640, 3) and images.dtype == np.uint8
            # All image bytes are covered by full-file SHA256 against the E1 manifest.
            assert np.ptp(images[0]) > 0 and np.ptp(images[-1]) > 0
        result['dataset'].append(dict(id=i, bytes=path.stat().st_size, sha256=digest))
        print('Dataset verified', i, flush=True)
    utils.set_seed(1)  # Exactly the official main() seed, BEFORE load_data.
    train, val, stats, is_sim = utils.load_data(str(data), 50, ['top'], 8, 8)
    split = dict(train=train.dataset.episode_ids.tolist(), validation=val.dataset.episode_ids.tolist())
    assert len(set(split['train']) & set(split['validation'])) == 0
    assert sorted(split['train']+split['validation']) == list(range(50))
    result['split'] = split
    result['normalization_scope'] = 'Official get_norm_stats uses all 50 trajectories, unchanged'
    assert is_sim
    for label, loader in [('train', train), ('validation', val)]:
        batch = next(iter(loader))
        assert [list(x.shape) for x in batch] == [[8,1,3,480,640],[8,14],[8,400,14],[8,400]]
        assert all(torch.isfinite(x).all() for x in batch)
        result[label+'_batch_shapes'] = [list(x.shape) for x in batch]
    history = ROOT/'E1_results'
    assert (history/'finish.txt').is_file()
    prior = json.loads((history/'reports/integrity_check.json').read_text())
    for name, rec in prior['models'].items():
        assert sha256(history/'models'/name/'policy_best.ckpt') == rec['checkpoint_sha256']
        assert (history/'models'/name/'dataset_stats.pkl').is_file()
    for path, expected in prior.get('files', {}).items():
        assert sha256(path) == expected, 'E1 integrity failed: '+path
    inventory = {}
    for path in sorted(history.rglob('*')):
        if path.is_symlink():
            inventory[str(path.relative_to(history))] = dict(symlink=os.readlink(str(path)))
        elif path.is_file():
            inventory[str(path.relative_to(history))] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    write_json(RUN/'provenance/e1_manifest_before.json', inventory)
    result['e1'] = dict(complete=True, inventory_files=len(inventory), finish_sha256=sha256(history/'finish.txt'))
    result.update(passed=True, completed_at=now())
    write_json(RUN/'reports/preflight.json', result)
    print('PREFLIGHT PASSED', flush=True)


if __name__ == '__main__':
    main()
