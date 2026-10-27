"""Paired full/top render test and old/new software renderer observations."""
import argparse
import json
import pickle
import sys
import time
from common import ROOT, RUN, now, write_json, activate_renderer
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--renderer', choices=['softpipe', 'llvmpipe', 'egl'], required=True)
    args = parser.parse_args()
    activate_renderer()
    import numpy as np
    import torch
    import h5py
    import random
    import sim_env
    from PIL import Image
    from render_controls import install_top_only, identity
    target = RUN/'tests'/('render_'+args.renderer)
    target.mkdir(exist_ok=False)
    with h5py.File(str(ROOT/'data/sim_transfer_cube_scripted/episode_0.hdf5'), 'r') as f:
        actions = f['action'][:5]
    poses = [[.03,.42,.05,1.,0.,0.,0.], [.17,.58,.05,1.,0.,0.,0.]]
    original = sim_env.BimanualViperXTask.get_observation
    timings, reference, frames, states = {}, [], [], []
    report = dict(passed=False, renderer=args.renderer, time=now(), poses=poses, steps=5, comparisons=[])
    for top_only in [False, True]:
        sim_env.BimanualViperXTask.get_observation = original
        if top_only:
            install_top_only(target)
        elapsed = []
        for index, pose in enumerate(poses):
            np.random.seed(1000); torch.manual_seed(1000); random.seed(1000)
            sim_env.BOX_POSE[0] = np.array(pose)
            env = sim_env.make_sim_env('sim_transfer_cube_scripted')
            try:
                # Initialize/compile shaders and warm all cameras before timed reset.
                env.reset()
                if 'gl_identity' not in report:
                    report['gl_identity'] = identity(env.physics)
                start = time.perf_counter()
                ts = env.reset()
                trajectory = []
                for step in range(6):
                    if step:
                        ts = env.step(actions[step-1])
                    obs = ts.observation
                    trajectory.append(dict(image=obs['images']['top'].copy(),
                        qpos=obs['qpos'].copy(), qvel=obs['qvel'].copy(),
                        env_state=obs['env_state'].copy(), full_state=env.physics.get_state().copy(),
                        reward=ts.reward, numpy_rng=pickle.dumps(np.random.get_state()),
                        python_rng=pickle.dumps(random.getstate()), torch_rng=torch.get_rng_state().numpy().copy()))
                elapsed.append(time.perf_counter()-start)
                assert identity(env.physics)['gl_error'] == 0
            finally:
                env.physics.free()
            if not top_only:
                reference.append(trajectory)
                for row in trajectory:
                    frames.append(row['image']); states.append(row['full_state'])
                Image.fromarray(trajectory[0]['image']).save(str(target/('pose%d.png'%index)))
            else:
                for a, b in zip(reference[index], trajectory):
                    for key in ['image','qpos','qvel','env_state','full_state','torch_rng']:
                        assert np.array_equal(a[key],b[key]), (index,key)
                    for key in ['reward','numpy_rng','python_rng']:
                        assert a[key] == b[key], (index,key)
                report['comparisons'].append(dict(pose=index, observations=6,
                    pixels_physics_rewards_numpy_python_torch_rng_exact=True))
            print('render', args.renderer, 'top_only', top_only, 'pose', index, 'seconds', elapsed[-1], flush=True)
        timings['top_only' if top_only else 'all_cameras'] = elapsed
    report['timings_seconds'] = timings
    report['camera_filter_speedup'] = sum(timings['all_cameras'])/sum(timings['top_only'])
    report['mean_top_step_seconds'] = sum(timings['top_only'])/12
    report.update(passed=True, context_released=True)
    np.savez_compressed(str(target/'comparison.npz'), images=np.array(frames), states=np.array(states))
    write_json(target/'validation.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
