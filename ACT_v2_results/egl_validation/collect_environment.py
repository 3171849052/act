"""Read-only diagnostics; writes only into this validation directory."""
import ctypes
import ctypes.util
import glob
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone, timedelta

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
PREFIX = Path('/HOME/sysu_qling/sysu_qling_3/HDD_POOL/anaconda3/envs/act-repro')


def run(name, argv, env=None, timeout=90):
    try:
        p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, env=env, timeout=timeout)
        content = '$ ' + repr(argv) + '\nexit_code=' + str(p.returncode) + '\n' + p.stdout
    except Exception as exc:
        content = repr(exc)
    (OUT / name).write_text(content)
    return content


def fingerprint():
    paths = sorted(PREFIX.glob('conda-meta/*.json')) + [PREFIX / 'conda-meta/history']
    paths += [ROOT / p for p in subprocess.check_output(
        ['git', 'ls-files'], cwd=str(ROOT), text=True).splitlines()]
    result = {}
    for p in paths:
        if p.is_file():
            result[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


if __name__ == '__main__':
    if '--verify' in sys.argv:
        before = json.loads((OUT / 'environment_fingerprint_before.json').read_text())
        after = fingerprint()
        changes = [k for k in set(before) | set(after) if before.get(k) != after.get(k)]
        (OUT / 'environment_integrity.json').write_text(json.dumps(
            {'tracked_files_and_conda_metadata_unchanged': not changes, 'changed': changes,
             'files_checked': len(before)}, indent=2))
        print('Integrity:', len(changes), 'changes')
        sys.exit(bool(changes))
    (OUT / 'environment_fingerprint_before.json').write_text(json.dumps(fingerprint(), indent=2))
    (OUT / 'collected_at.txt').write_text(datetime.now(timezone(timedelta(hours=8))).isoformat())
    run('nvidia_smi.txt', ['nvidia-smi'])
    run('nvidia_smi_q.txt', ['nvidia-smi', '-q'])
    run('gpu_identity.csv', ['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total,pci.bus_id', '--format=csv'])
    for src, dest in [('/proc/1/cgroup', 'cgroup.txt'), ('/proc/self/mountinfo', 'mountinfo.txt'),
                      ('/etc/os-release', 'os_release.txt')]:
        (OUT / dest).write_text(Path(src).read_text())
    relevant = {k: v for k, v in os.environ.items() if any(s in k.upper() for s in
        ['NVIDIA', 'CUDA', 'CONTAINER', 'APPTAINER', 'SINGULARITY', 'CONDA',
         'EGL', 'OPENGL', 'MUJOCO', 'LD_LIBRARY', 'DISPLAY'])}
    devices = {}
    for p in glob.glob('/dev/nvidia*') + glob.glob('/dev/dri/*'):
        devices[p] = {'readable': os.access(p, os.R_OK), 'writable': os.access(p, os.W_OK),
                      'realpath': os.path.realpath(p)}
    info = {'environment': relevant, 'uid': os.getuid(), 'gid': os.getgid(),
            'markers': {p: os.path.exists(p) for p in ['/.dockerenv', '/run/.containerenv',
                '/var/run/docker.sock', '/run/podman/podman.sock', '/dev/dri']},
            'commands': {x: shutil.which(x) for x in ['docker', 'podman', 'apptainer',
                'singularity', 'nvidia-container-cli', 'kubectl', 'ctr', 'crictl']},
            'devices': devices}
    (OUT / 'runtime_environment.json').write_text(json.dumps(info, indent=2))
    run('device_nodes.txt', ['bash', '-c', 'ls -l /dev/nvidia* /dev/dri/*'])
    run('ldconfig.txt', ['ldconfig', '-p'])
    patterns = ['libEGL*', 'libGLdispatch*', 'libOpenGL*', 'libGLX_nvidia*', 'libnvidia-egl*']
    args = ['find', '/usr', '/opt', '/lib', '/run/nvidia', str(PREFIX), '-name', patterns[0]]
    for pattern in patterns[1:]:
        args += ['-o', '-name', pattern]
    args += ['-o', '-path', '*/egl_vendor.d/*.json']
    found = run('library_search.txt', args, timeout=120)
    libraries = sorted({p for p in found.splitlines() if p.startswith('/') and os.path.isfile(p)})
    vendor_dirs = ['/usr/share/glvnd/egl_vendor.d', '/etc/glvnd/egl_vendor.d',
                   str(PREFIX / 'share/glvnd/egl_vendor.d')]
    vendors = {}
    for directory in vendor_dirs:
        vendors[directory] = {'exists': os.path.isdir(directory), 'files': {}}
        for p in glob.glob(directory + '/*.json'):
            vendors[directory]['files'][p] = Path(p).read_text()
    (OUT / 'egl_vendors.json').write_text(json.dumps(vendors, indent=2))
    deps = []
    seen = set()
    for p in libraries:
        real = os.path.realpath(p)
        if real not in seen and '.so' in p:
            seen.add(real)
            proc = subprocess.run(['ldd', p], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            deps.append({'path': p, 'readlink': real, 'exit_code': proc.returncode, 'ldd': proc.stdout})
    (OUT / 'library_dependencies.json').write_text(json.dumps(deps, indent=2))
    clean = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                 LD_LIBRARY_PATH=str(PREFIX / 'lib/act-graphics'))
    clean.pop('PYTHONPATH', None)
    code = '''import sys, json, ctypes, ctypes.util, importlib.metadata as m
r = {'python': sys.version, 'executable': sys.executable, 'packages': {}, 'libraries': {}}
for p in ['mujoco','dm-control','PyOpenGL','PyOpenGL-accelerate','glfw','numpy','Pillow','torch']:
 try: r['packages'][p] = m.version(p)
 except m.PackageNotFoundError: r['packages'][p] = None
for name in ['EGL','EGL_nvidia','GLdispatch','OpenGL','GLX_nvidia','OSMesa']:
 entry = {'find_library': ctypes.util.find_library(name)}
 for path in [entry['find_library'], 'lib'+name+'.so', 'lib'+name+'.so.1', 'lib'+name+'.so.0']:
  if not path: continue
  try: lib=ctypes.CDLL(path); entry[path]={'loaded':True, 'name':lib._name}
  except OSError as e: entry[path]={'loaded':False, 'error':str(e)}
 r['libraries'][name] = entry
print(json.dumps(r, indent=2))
'''
    run('python_dependencies_and_loading.txt', [str(PREFIX / 'bin/python'), '-c', code], env=clean)
    print('Diagnostics saved:', OUT)
