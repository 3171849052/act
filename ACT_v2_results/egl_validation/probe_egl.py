"""Fresh-process dm_control EGL probe. No software renderer is accepted."""
import os
if __name__ == '__main__':
    os.environ.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
                      MUJOCO_EGL_DEVICE_ID='0', MPLBACKEND='Agg')

import ctypes
import ctypes.util
from datetime import datetime, timezone, timedelta
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time
import traceback

OUT = Path(__file__).resolve().parent


def gl_identity(physics):
    """Query on dm_control's actual render thread with its own context bound."""
    from OpenGL import GL
    def query():
        result = {}
        for key in ['GL_VENDOR', 'GL_RENDERER', 'GL_VERSION']:
            value = GL.glGetString(getattr(GL, key))
            if not value:
                raise RuntimeError('No valid current context: ' + key)
            result[key] = value.decode('utf-8', errors='replace')
        result['gl_error'] = int(GL.glGetError())
        if result['gl_error']:
            raise RuntimeError('OpenGL error: ' + str(result))
        return result
    with physics.contexts.gl.make_current() as context:
        result = context.call(query)
    result['context_class'] = type(physics.contexts.gl).__module__ + '.' + type(physics.contexts.gl).__name__
    return result


def require_nvidia(identity):
    vendor = identity['GL_VENDOR'].lower()
    renderer = identity['GL_RENDERER'].lower()
    if 'nvidia' not in vendor or any(x in renderer for x in ['llvmpipe', 'softpipe', 'swrast', 'software']):
        raise RuntimeError('NVIDIA GPU renderer required: ' + repr(identity))
    if 'egl' not in identity['context_class'].lower():
        raise RuntimeError('dm_control context is not EGL: ' + repr(identity))


def main():
    result = {'passed': False, 'stage': 'library_diagnostics', 'gl_identity': None,
              'environment': {k: os.environ.get(k) for k in ['MUJOCO_GL', 'PYOPENGL_PLATFORM',
                  'MUJOCO_EGL_DEVICE_ID', 'MPLBACKEND', 'LD_LIBRARY_PATH']},
              'time': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'versions': {p: metadata.version(p) for p in ['mujoco', 'dm-control', 'PyOpenGL']}}
    start = time.perf_counter()
    physics = None
    try:
        result['egl_find_library'] = ctypes.util.find_library('EGL')
        try:
            lib = ctypes.CDLL('libEGL.so.1')
            result['egl_cdll'] = lib._name
        except OSError as exc:
            result['egl_cdll_error'] = str(exc)
        result['stage'] = 'import_mujoco'
        import mujoco
        result['stage'] = 'import_dm_control_mujoco'
        from dm_control import mujoco as dm_mujoco
        result['stage'] = 'create_physics'
        physics = dm_mujoco.Physics.from_xml_string('''<mujoco><visual><global offwidth="640" offheight="480"/></visual>
<worldbody><light pos="0 0 3"/><geom type="plane" size="2 2 .1" rgba=".3 .4 .5 1"/>
<body pos="0 0 .2"><freejoint/><geom type="box" size=".1 .1 .1" rgba="1 .1 .1 1"/></body>
<camera name="probe" pos="1 -1 1" xyaxes=".707 .707 0 -.408 .408 .816"/></worldbody></mujoco>''')
        result['stage'] = 'create_and_bind_dm_control_egl_context'
        result['gl_identity'] = gl_identity(physics)
        require_nvidia(result['gl_identity'])
        result['stage'] = 'render_rgb'
        t = time.perf_counter()
        rgb = physics.render(height=480, width=640, camera_id='probe')
        result['render_seconds'] = time.perf_counter() - t
        from PIL import Image
        assert rgb.shape == (480, 640, 3) and str(rgb.dtype) == 'uint8'
        Image.fromarray(rgb).save(str(OUT / 'egl_probe_rgb.png'))
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
        result['elapsed_seconds'] = time.perf_counter() - start
        (OUT / 'egl_probe_result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
