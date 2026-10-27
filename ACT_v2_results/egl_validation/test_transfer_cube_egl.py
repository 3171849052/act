"""Short rendering validation only: no policy, training, or model evaluation.

Uses installed dm_control 1.0.14 APIs and the repository's unmodified environment.
The OSMesa mode supplies a baseline; it can never count as an EGL pass.
"""
import argparse
import os
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['egl', 'osmesa'], default='egl')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--steps', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 2 or args.steps < 2:
        parser.error('At least two repeats and two steps required')
    os.environ.update(MUJOCO_GL=args.backend, PYOPENGL_PLATFORM=args.backend,
                      MUJOCO_EGL_DEVICE_ID='0', MPLBACKEND='Agg')
    # All third-party imports happen AFTER backend environment selection.
    import json
    import time
    import traceback
    result = {'passed': False, 'backend': args.backend, 'stage': 'preflight',
              'repeats': args.repeats, 'steps_per_repeat': args.steps,
              'fixed_box_pose': [0.15, 0.5, 0.05, 1., 0., 0., 0.],
              'camera': 'top', 'image_shape': [480, 640, 3], 'image_channels': 'RGB',
              'timing_note': 'reset/step include original ACT top,angle,vis observations; render is one additional top frame; warmup excluded'}
    physics = None
    try:
        if args.backend == 'egl':
            probe = json.loads((OUT / 'egl_probe_result.json').read_text())
            if not probe.get('passed'):
                raise RuntimeError('BLOCKED: NVIDIA EGL probe has not passed; no Transfer Cube GPU run attempted')
        result['stage'] = 'imports'
        sys.path.insert(0, str(ROOT))
        import numpy as np
        from PIL import Image
        import sim_env
        from probe_egl import gl_identity, require_nvidia
        np.random.seed(0)
        sim_env.BOX_POSE[0] = np.array(result['fixed_box_pose'], dtype=np.float64)
        result['stage'] = 'make_sim_env'
        env = sim_env.make_sim_env('sim_transfer_cube_scripted')
        physics = env.physics
        result['stage'] = 'initial_reset'
        initial = env.reset()
        result['gl_identity'] = gl_identity(physics)
        if args.backend == 'egl':
            require_nvidia(result['gl_identity'])
        else:
            if 'osmesa' not in result['gl_identity']['context_class'].lower():
                raise RuntimeError('Baseline is not OSMesa')
        action_base = initial.observation['qpos'].copy()
        actions = np.repeat(action_base[None, :], args.steps, axis=0)
        actions[:, [6, 13]] = np.clip(actions[:, [6, 13]], 0., 1.)
        actions[:, 0] += .005 * np.sin(np.arange(args.steps) * .3)
        actions[:, 7] -= .005 * np.sin(np.arange(args.steps) * .3)
        # Validate the actual 16-actuator commands, including gripper conversion.
        from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN as unnormalize
        for action in actions:
            left, right = unnormalize(action[6]), unnormalize(action[13])
            control = np.r_[action[:6], left, -left, action[7:13], right, -right]
            limited = physics.model.actuator_ctrllimited.astype(bool)
            bounds = physics.model.actuator_ctrlrange
            if not (np.isfinite(control).all() and
                    np.all(control[limited] >= bounds[limited, 0]) and
                    np.all(control[limited] <= bounds[limited, 1])):
                raise RuntimeError('Action exceeds actuator control limits')
        result['actions_valid'] = True
        result['stage'] = 'warmup'
        for _ in range(2):
            env.reset()
            for action in actions[:2]:
                env.step(action)
            physics.render(height=480, width=640, camera_id='top')
        timings = {'reset': [], 'step': [], 'render_top': []}
        snapshots = {k: [] for k in ['qpos', 'qvel', 'obs_qpos', 'obs_qvel',
                     'env_state', 'reward', 'step_type', 'time', 'ctrl', 'images']}

        def capture(timestep):
            obs = timestep.observation
            rgb = obs['images']['top']
            if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                raise RuntimeError('Invalid RGB shape/dtype')
            for key, value in [('qpos', physics.data.qpos), ('qvel', physics.data.qvel),
                               ('ctrl', physics.data.ctrl), ('obs_qpos', obs['qpos']),
                               ('obs_qvel', obs['qvel']), ('env_state', obs['env_state'])]:
                if not np.isfinite(value).all():
                    raise RuntimeError('Nonfinite state: ' + key)
                snapshots[key].append(np.array(value, copy=True))
            reward = -1. if timestep.reward is None else float(timestep.reward)
            if not np.isfinite(reward):
                raise RuntimeError('Nonfinite reward')
            snapshots['reward'].append(reward)  # -1 encodes reset reward=None.
            snapshots['step_type'].append(int(timestep.step_type))
            snapshots['time'].append(float(physics.data.time))
            snapshots['images'].append(rgb.copy())

        result['stage'] = 'measured_trials'
        start = time.perf_counter()
        for trial in range(args.repeats):
            t = time.perf_counter()
            timestep = env.reset()
            timings['reset'].append(time.perf_counter() - t)
            capture(timestep)
            if trial == 0:
                Image.fromarray(timestep.observation['images']['top']).save(str(OUT / (args.backend + '_transfer_top_reset.png')))
            for action in actions:
                t = time.perf_counter()
                timestep = env.step(action)
                timings['step'].append(time.perf_counter() - t)
                capture(timestep)
                t = time.perf_counter()
                rgb = physics.render(height=480, width=640, camera_id='top')
                timings['render_top'].append(time.perf_counter() - t)
                assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
        elapsed = time.perf_counter() - start
        result['measured_wall_seconds'] = elapsed
        result['gl_identity_after'] = gl_identity(physics)
        if args.backend == 'egl':
            require_nvidia(result['gl_identity_after'])
        warnings = physics.data.warning.number.copy()
        result['mujoco_warning_counts'] = warnings.tolist()
        if np.any(warnings):
            raise RuntimeError('MuJoCo issued simulation warnings')
        Image.fromarray(timestep.observation['images']['top']).save(str(OUT / (args.backend + '_transfer_top_final.png')))
        np.savez_compressed(str(OUT / (args.backend + '_trajectory.npz')),
                            actions=actions, **{k: np.array(v) for k, v in snapshots.items()})
        result['timings_seconds'] = timings
        result['statistics'] = {k: {'mean_seconds': float(np.mean(v)),
            'std_seconds': float(np.std(v)), 'min_seconds': float(np.min(v)),
            'max_seconds': float(np.max(v)), 'samples': len(v)} for k, v in timings.items()}
        count = args.repeats * args.steps
        result['throughput'] = {'steps_per_wall_second': count / elapsed,
            'env_steps_per_second': count / sum(timings['step']),
            'top_renders_per_second': count / sum(timings['render_top'])}
        result['captured_frames'] = len(snapshots['images'])
        result['stage'] = 'complete'
        result['passed'] = True
    except Exception:
        result['error'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        if physics is not None:
            try:
                physics.free()
                result['context_released'] = True
            except Exception:
                result['passed'] = False
                result['cleanup_error'] = traceback.format_exc()
        path = OUT / (args.backend + '_transfer_result.json')
        path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
