"""Run matching CPU/GPU rendering tests in isolated processes, fail closed.

Writes no success marker. A blocked GPU leaves speedup and comparisons null.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

OUT = Path(__file__).resolve().parent


def run_backend(backend):
    env = dict(os.environ, MUJOCO_GL=backend, PYOPENGL_PLATFORM=backend,
               MUJOCO_EGL_DEVICE_ID='0', MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
    filename = 'gpu_render_test.log' if backend == 'egl' else 'osmesa_render_test.log'
    with (OUT / filename).open('w') as log:
        try:
            process = subprocess.run([sys.executable, '-u', str(OUT / 'test_transfer_cube_egl.py'),
                '--backend', backend, '--repeats', '3', '--steps', '5'], env=env,
                stdout=log, stderr=subprocess.STDOUT, timeout=240, cwd=str(OUT.parent.parent))
        except subprocess.TimeoutExpired:
            return {'passed': False, 'error': 'Own test subprocess timed out after 240s'}
    path = OUT / (backend + '_transfer_result.json')
    if process.returncode or not path.exists():
        return {'passed': False, 'exit_code': process.returncode,
                'details': json.loads(path.read_text()) if path.exists() else None}
    return json.loads(path.read_text())


def main():
    result = {'status': 'running', 'nvidia_gpu_enabled': False, 'speedup_top_render': None,
              'physics_comparison': None, 'image_comparison': None, 'egl': None,
              'measurement': '3 repeats x 5 steps, after 2 warmup resets x 2 steps; unmodified ACT all-camera reset/step; standalone render measures top 480x640 RGB'}
    try:
        result['osmesa'] = run_backend('osmesa')
        # The GPU worker has a preflight gate and will not import graphics if the probe failed.
        result['egl'] = run_backend('egl')
        if not result['osmesa']['passed'] or not result['egl']['passed']:
            result['status'] = 'blocked'
            result['reason'] = 'Both NVIDIA EGL and OSMesa must pass before comparisons or speedup exist'
        else:
            import numpy as np
            with np.load(str(OUT / 'osmesa_trajectory.npz')) as cpu, np.load(str(OUT / 'egl_trajectory.npz')) as gpu:
                states = {}
                for key in ['actions', 'qpos', 'qvel', 'obs_qpos', 'obs_qvel', 'env_state', 'reward', 'step_type', 'time', 'ctrl']:
                    states[key] = {'max_abs_difference': float(np.max(np.abs(cpu[key] - gpu[key]))),
                                   'allclose': bool(np.allclose(cpu[key], gpu[key], atol=1e-10, rtol=1e-10))}
                result['physics_comparison'] = states
                c, g = cpu['images'], gpu['images']
                assert c.shape == g.shape and c.dtype == g.dtype == np.uint8
                delta = np.abs(c.astype(np.float64) - g.astype(np.float64))
                per_pixel = np.max(delta, axis=-1)
                mse = float(np.mean(delta ** 2))
                result['image_comparison'] = {'shape': list(c.shape), 'dtype': str(c.dtype), 'channels': 'RGB',
                    'mean_abs_difference': float(delta.mean()), 'max_abs_difference': float(delta.max()),
                    'rmse': float(np.sqrt(mse)), 'channel_mean_abs_difference': delta.mean(axis=(0, 1, 2)).tolist(),
                    'fraction_pixels_any_difference': float(np.mean(per_pixel > 0)),
                    'fraction_pixels_difference_gt_10': float(np.mean(per_pixel > 10)),
                    'p99_abs_difference': float(np.percentile(delta, 99)),
                    'visual_review_required': True}
            result['nvidia_gpu_enabled'] = True
            result['speedup_top_render'] = (result['osmesa']['statistics']['render_top']['mean_seconds'] /
                                             result['egl']['statistics']['render_top']['mean_seconds'])
            result['physics_consistent'] = all(x['allclose'] for x in states.values())
            result['status'] = 'measured_pending_visual_review' if result['physics_consistent'] else 'physics_mismatch'
    except Exception:
        result['status'] = 'error'
        result['error'] = traceback.format_exc()
        traceback.print_exc()
    (OUT / 'benchmark_results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ['osmesa', 'egl']}, indent=2), flush=True)
    return 0 if result['status'] == 'measured_pending_visual_review' else 1


if __name__ == '__main__':
    sys.exit(main())
