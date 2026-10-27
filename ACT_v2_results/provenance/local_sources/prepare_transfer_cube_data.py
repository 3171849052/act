#!/usr/bin/env python3
"""Safely prepare local Transfer Cube ZIPs. Never alter source ZIPs or HDF5s.

Run with an existing Python containing h5py/numpy; torch/IPython are needed
only for the official ACT loading smoke test. No dependencies are installed.
All archive members are streamed through zipfile's CRC verification. Duplicate
episodes are compared by SHA256 before any dataset is extracted.
"""
import argparse
import ast
import contextlib
import csv
import ctypes
import errno
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import sys
import time
import traceback
import uuid
import zipfile
from datetime import datetime, timezone, timedelta

CHUNK = 8 * 1024 * 1024
RESERVE = 1024 ** 3
KINDS = ('human', 'scripted')
EPISODE = re.compile(r'episode_(0|[1-9][0-9]*)\.hdf5\Z')
FIELDS = ['filename', 'size_bytes', 'sha256', 'trajectory_length', 'image_shape',
          'qpos_shape', 'qvel_shape', 'action_shape', 'sim', 'validation_result',
          'source_archives', 'source_members', 'error']


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def log(message):
    print(f'[{now()}] {message}', flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.part')
    with temporary.open('x', encoding='utf-8', newline='') as output:
        output.write(value)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)  # reports only, never dataset files


def save_csv(path, fields, rows):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def space(path, required):
    available = shutil.disk_usage(path).free
    if available < required + RESERVE:
        raise RuntimeError(f'磁盘空间不足：可用 {available}，需要 {required} + {RESERVE} 字节安全余量')
    return available


def publish_no_replace(source, target):
    """Atomic Linux rename with RENAME_NOREPLACE; never overwrite a target."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    if rename is not None:
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                           ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1) == 0:
            return
        code = ctypes.get_errno()
        if code not in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
            raise OSError(code, os.strerror(code), str(target))
    # Portable no-clobber atomic publication; hard link creates no second copy.
    os.link(source, target, follow_symlinks=False)
    source.unlink()


def check_member(info, dataset):
    name = info.filename
    parts = PurePosixPath(name).parts
    if (not name or '\x00' in name or '\\' in name or name.startswith('/')
            or any(p in ('..', '.') or ':' in p for p in parts)):
        raise ValueError(f'不安全的 ZIP 路径：{name!r}')
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
        raise ValueError(f'ZIP 含符号链接/特殊文件：{name}')
    if info.flag_bits & 1:
        raise ValueError(f'不支持加密 ZIP 成员：{name}')
    base = PurePosixPath(name).name
    if info.is_dir() or '__MACOSX' in parts or base == '.DS_Store':
        return None
    if base.lower().endswith(('.hdf5', '.h5')):
        match = EPISODE.fullmatch(base)
        if not match:
            raise ValueError(f'HDF5 文件名不符合白名单：{name}')
        other = 'sim_transfer_cube_' + ('scripted' if dataset.endswith('human') else 'human')
        if other in parts:
            raise ValueError(f'ZIP 文件名与内部数据集类型不符：{name}')
        return int(match.group(1))
    return None


def check_single_disk(path):
    with path.open('rb') as source:
        source.seek(max(0, path.stat().st_size - 65557))
        tail = source.read()
    # Locate an EOCD whose comment ends exactly at EOF (signature may be in comment).
    positions = [m.start() for m in re.finditer(re.escape(b'PK\x05\x06'), tail)]
    for offset in reversed(positions):
        if len(tail) - offset < 22:
            continue
        fields = struct.unpack_from('<4s4H2LH', tail, offset)
        if offset + 22 + fields[-1] != len(tail):
            continue
        _, disk, central_disk, entries_disk, entries_total, *_ = fields
        if disk or central_disk or entries_disk != entries_total:
            raise ValueError('真正多磁盘/分卷 ZIP，需要原始全部分卷后另行处理')
        return
    raise ValueError('找不到有效 ZIP 结束记录，可能损坏/截断/缺少分卷')


def validate_hdf5(path):
    import h5py
    import numpy as np
    result = {}
    expected = {'/observations/images/top': (400, 480, 640, 3),
                '/observations/qpos': (400, 14), '/observations/qvel': (400, 14),
                '/action': (400, 14)}
    keys = ['image_shape', 'qpos_shape', 'qvel_shape', 'action_shape']
    with h5py.File(path, 'r') as root:
        if 'sim' not in root.attrs:
            raise ValueError('缺少根属性 sim')
        sim = np.asarray(root.attrs['sim'])
        if sim.shape != () or sim.dtype.kind not in 'biu' or sim.item() != 1:
            raise ValueError(f'sim 应为仿真标识 True/1，实际 {sim!r}')
        result['sim'] = str(sim.item())
        for (name, shape), key in zip(expected.items(), keys):
            node = root
            for component in name.strip('/').split('/'):
                if not isinstance(node.get(component, getlink=True), h5py.HardLink):
                    raise ValueError(f'字段缺失或包含软/外部链接：{name}')
                node = node[component]
            if not isinstance(node, h5py.Dataset):
                raise ValueError(f'{name} 不是 dataset')
            if node.is_virtual or node.external:
                raise ValueError(f'{name} 依赖外部/虚拟存储')
            result[key] = str(tuple(node.shape))
            if tuple(node.shape) != shape:
                raise ValueError(f'{name} 形状 {node.shape} != {shape}')
            image = name.endswith('/top')
            if image and node.dtype != np.dtype('uint8'):
                raise ValueError(f'图像 dtype {node.dtype} != uint8')
            if not image and node.dtype.kind not in 'fiu':
                raise ValueError(f'{name} dtype {node.dtype} 非有效实数类型')
            # Read ALL image frames and numeric samples with bounded memory.
            for start in range(0, shape[0], 8 if image else 64):
                block = node[start:start + (8 if image else 64)]
                if not np.isfinite(block).all():
                    raise ValueError(f'{name} 在 {start} 附近含 NaN/Inf')
        result['trajectory_length'] = 400
    result['validation_result'] = '已成功验证'
    return result


def loading_test(root, datasets):
    import numpy as np
    import torch
    import utils
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    results = {}
    for kind in datasets:
        try:
            config = __import__('constants').SIM_TASK_CONFIGS['sim_transfer_cube_' + kind]
            directory = Path(config['dataset_dir'])
            assert directory.resolve() == (root / 'data' / ('sim_transfer_cube_' + kind)).resolve()
            assert config['num_episodes'] == 50 and config['episode_len'] == 400
            assert config['camera_names'] == ['top']
            train, val, stats, sim = utils.load_data(str(directory), 50, ['top'], 2, 2)
            assert bool(sim)
            for split, loader in [('train', train), ('val', val)]:
                batch = next(iter(loader))
                expected = [(2, 1, 3, 480, 640), (2, 14), (2, 400, 14), (2, 400)]
                for tensor, shape in zip(batch, expected):
                    assert tuple(tensor.shape) == shape, (tensor.shape, shape)
                    assert tensor.device.type == 'cpu'
                    assert torch.isfinite(tensor).all()
                assert batch[0].dtype == torch.float32
                assert batch[1].dtype == batch[2].dtype == torch.float32
                assert batch[3].dtype == torch.bool
                assert 0 <= batch[0].min() <= batch[0].max() <= 1
                assert not batch[3][:, 0].any()
                assert not ((~batch[3][:, 1:]) & batch[3][:, :-1]).any()
                print(kind, split, 'shapes=', [tuple(t.shape) for t in batch],
                      'dtypes=', [str(t.dtype) for t in batch], 'CPU PASS', flush=True)
            # Actual official __getitem__ on a specified episode, in addition to batches.
            specified = utils.EpisodicDataset([0], str(directory), ['top'], stats)[0]
            assert specified[0].shape == (1, 3, 480, 640)
            assert all(np.isfinite(value).all() for value in stats.values())
            results[kind] = '已成功验证：官方 load_data，训练/验证各一批，EpisodicDataset episode_0，CPU'
        except Exception:
            results[kind] = '验证失败：' + traceback.format_exc()
            print(results[kind], flush=True)
    return results


def configure(root, states, report):
    present = [s for s in states.values() if s['present']]
    if not present or not all(s['validated'] for s in present):
        return '尚未修改：至少一个现有数据集未通过校验'
    path = root / 'constants.py'
    text = path.read_text(encoding='utf-8')
    tree = ast.parse(text)
    assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'DATA_DIR' for t in n.targets)]
    if len(assignments) != 1:
        raise ValueError('无法唯一定位 DATA_DIR，未修改 constants.py')
    assignment = assignments[0]
    if assignment.lineno != assignment.end_lineno:
        raise ValueError('DATA_DIR 非单行赋值，未修改 constants.py')
    try:
        value = ast.literal_eval(assignment.value)
        if isinstance(value, str) and Path(value).resolve() == (root / 'data').resolve():
            return '无需修改：DATA_DIR 已正确'
    except (ValueError, TypeError):
        pass
    backup = report / ('constants.py.before.' + datetime.now().strftime('%Y%m%dT%H%M%S')
                       + '.' + uuid.uuid4().hex[:8] + '.bak')
    with backup.open('xb') as destination:
        destination.write(path.read_bytes())
        destination.flush()
        os.fsync(destination.fileno())
    lines = text.splitlines(keepends=True)
    index = assignment.lineno - 1
    line = lines[index]
    value_start = assignment.value.col_offset
    value_end = assignment.value.end_col_offset
    # AST offsets are byte offsets. Existing assignment is ASCII in this repository.
    encoded = line.encode('utf-8')
    lines[index] = (encoded[:value_start] + repr(str(root / 'data')).encode('utf-8')
                    + encoded[value_end:]).decode('utf-8')
    atomic_text(path, ''.join(lines))
    return f'已修改：只替换 DATA_DIR；备份 {backup}'


def run(root):
    data = root / 'data'
    reports = root / 'data_preparation_reports'
    reports.mkdir(exist_ok=True)
    states = {k: {'present': False, 'zips': [], 'errors': [], 'validated': False,
                  'rows': [], 'members': [], 'new_files': 0, 'reused_files': 0,
                  'loading': '因缺失数据无法执行'} for k in KINDS}
    inventory = []
    initial_free = shutil.disk_usage(data).free
    initial_zips = sorted(p for p in data.iterdir() if p.suffix.lower() == '.zip')
    def snapshot():
        payload = {'timestamp_shanghai': now(), 'root': str(root), 'python': sys.executable,
                   'initial_available_bytes': initial_free, 'zip_inventory': inventory,
                   'datasets': states}
        atomic_text(reports / 'preparation_state.json', json.dumps(payload, ensure_ascii=False, indent=2))
        atomic_text(reports / 'zip_sha256.txt', ''.join(f"{z['sha256']}  {z['path']}\n" for z in inventory))
        save_csv(reports / 'zip_inventory.csv', ['filename', 'size_bytes', 'sha256', 'kind', 'crc', 'error'], inventory)
        for kind, state in states.items():
            if state['present']:
                save_csv(reports / (kind + '_manifest.csv'), FIELDS, state['rows'])
        save_csv(reports / 'archive_members.csv', ['kind', 'archive', 'member', 'episode',
                  'size_bytes', 'compressed_bytes', 'crc32', 'sha256'],
                 [member for state in states.values() for member in state['members']])

    for kind in KINDS:
        state = states[kind]
        name = 'sim_transfer_cube_' + kind
        state['present'] = (data / name).exists() or any(p.name.startswith(name) for p in initial_zips)
        state['loading'] = '尚未验证' if state['present'] else '因缺失数据无法执行'
    log(f'ROOT={root}; Python={sys.executable}; free={initial_free}; ZIPs={len(initial_zips)}')
    # Full inventory and source hash are retained even for unreadable ZIPs.
    for path in initial_zips:
        kind = next((k for k in KINDS if path.name.startswith('sim_transfer_cube_' + k)), '')
        entry = {'filename': path.name, 'path': str(path), 'size_bytes': path.stat().st_size,
                 'kind': kind, 'sha256': sha256(path), 'crc': '尚未验证', 'error': ''}
        inventory.append(entry)
        if not kind:
            entry['crc'] = '不属于 Transfer Cube，未检测'
            continue
        state = states[kind]
        state['zips'].append(path.name)
        try:
            if path.is_symlink():
                raise ValueError('ZIP 为符号链接，拒绝处理')
            check_single_disk(path)
            with zipfile.ZipFile(path, 'r') as archive:
                for info in archive.infolist():
                    episode = check_member(info, 'sim_transfer_cube_' + kind)
                    if info.is_dir():
                        continue
                    digest = hashlib.sha256()
                    count = 0
                    with archive.open(info, 'r') as source:
                        while block := source.read(CHUNK):
                            count += len(block)
                            if episode is not None:
                                digest.update(block)
                    if count != info.file_size:
                        raise ValueError(f'成员长度错误：{info.filename}')
                    if episode is not None:
                        state['members'].append({'kind': kind, 'archive': path.name,
                            'member': info.filename, 'episode': episode,
                            'size_bytes': info.file_size, 'compressed_bytes': info.compress_size,
                            'crc32': f'{info.CRC:08x}', 'sha256': digest.hexdigest()})
            entry['crc'] = '已成功验证：独立普通 ZIP，所有文件流式读取/CRC 通过'
            log(f'CRC/SHA256 PASS {path.name}')
        except Exception as error:
            entry['crc'] = '验证失败'
            entry['error'] = f'{type(error).__name__}: {error}'
            state['errors'].append(f'{path.name}: {entry["error"]}')
            log(f'ZIP FAIL {path.name}: {error}')
        snapshot()

    # Verify duplicates, index continuity, existing targets and capacity BEFORE extraction.
    plans = {}
    for kind, state in states.items():
        if not state['present']:
            continue
        by_id = {}
        for member in state['members']:
            by_id.setdefault(member['episode'], []).append(member)
        target_dir = data / ('sim_transfer_cube_' + kind)
        if target_dir.is_symlink():
            state['errors'].append('目标数据集目录为符号链接，拒绝处理')
        existing = {}
        if target_dir.exists() and not target_dir.is_symlink():
            if not target_dir.is_dir():
                state['errors'].append('目标数据集路径不是目录')
            else:
                for path in target_dir.iterdir():
                    if path.name.endswith('.part'):
                        state.setdefault('warnings', []).append(f'保留中断临时文件：{path.name}')
                    elif path.suffix.lower() in ('.hdf5', '.h5'):
                        match = EPISODE.fullmatch(path.name)
                        if not match or path.is_symlink() or not path.is_file():
                            state['errors'].append(f'现有 HDF5 路径异常：{path.name}')
                        else:
                            existing[int(match.group(1))] = path
                    elif path.is_dir():
                        state['errors'].append(f'目标目录包含嵌套目录：{path.name}，需人工检查')
        ids = set(by_id) | set(existing)
        if ids != set(range(50)):
            state['errors'].append(f'编号不完整：缺失 {sorted(set(range(50))-ids)}，超范围 {sorted(ids-set(range(50)))}')
        selected = {}
        duplicate_ids = []
        for episode in sorted(ids):
            members = by_id.get(episode, [])
            filename = f'episode_{episode}.hdf5'
            row = {'filename': filename, 'source_archives': ' | '.join(m['archive'] for m in members),
                   'source_members': ' | '.join(m['member'] for m in members), 'validation_result': '尚未验证'}
            state['rows'].append(row)
            hashes = {m['sha256'] for m in members}
            if members:
                row.update(size_bytes=members[0]['size_bytes'], sha256=members[0]['sha256'])
                selected[episode] = members[0]
            if len(members) > 1:
                duplicate_ids.append(episode)
            if len(hashes) > 1:
                message = f'{filename} 重名内容冲突：' + ' | '.join(f"{m['archive']}:{m['sha256']}" for m in members)
                state['errors'].append(message)
                row.update(validation_result='验证失败', error=message)
            if episode in existing:
                path = existing[episode]
                actual = sha256(path)
                if hashes and actual not in hashes:
                    message = f'现有目标 {path} 与 ZIP 内容哈希冲突，禁止覆盖'
                    state['errors'].append(message)
                    row.update(validation_result='验证失败', error=message)
                elif len(hashes) <= 1:
                    row.update(size_bytes=path.stat().st_size, sha256=actual)
        state['duplicate_episode_ids'] = duplicate_ids
        state['expected_unique_bytes'] = sum(int(r.get('size_bytes', 0)) for r in state['rows'])
        state['additional_required_bytes'] = sum(m['size_bytes'] for i, m in selected.items() if i not in existing)
        state['existing_episode_ids'] = sorted(existing)
        plans[kind] = (selected, existing)
        if state['errors']:
            state['loading'] = '因数据校验失败未执行'
            log(f'{kind} STOP: {state["errors"]}')
        else:
            log(f'{kind}: {len(ids)} unique episodes, {len(duplicate_ids)} duplicate IDs (SHA256 identical), '
                f'additional={state["additional_required_bytes"]} bytes')
    total_required = sum(s.get('additional_required_bytes', 0) for s in states.values() if not s['errors'])
    try:
        space(data, total_required)
        log(f'CAPACITY PASS: require={total_required} + reserve={RESERVE}; available={shutil.disk_usage(data).free}')
    except Exception as error:
        for state in states.values():
            if state['present'] and not state['errors']:
                state['errors'].append(str(error))
    snapshot()

    try:
        import h5py
        import numpy
        versions = f'Python={sys.version}; h5py={h5py.__version__}; numpy={numpy.__version__}'
    except Exception as error:
        versions = f'缺少验证依赖：{type(error).__name__}: {error}'
        for state in states.values():
            if state['present']:
                state['errors'].append(versions)

    for kind, state in states.items():
        if not state['present'] or state['errors']:
            continue
        target_dir = data / ('sim_transfer_cube_' + kind)
        selected, existing = plans[kind]
        try:
            # Validate every existing target before writing any new episode.
            for episode, path in sorted(existing.items()):
                row = next(r for r in state['rows'] if r['filename'] == path.name)
                row.update(validate_hdf5(path))
                state['reused_files'] += 1
            target_dir.mkdir(exist_ok=True)
            for row in state['rows']:
                episode = int(EPISODE.fullmatch(row['filename']).group(1))
                if episode in existing:
                    continue
                member = selected[episode]
                target = target_dir / row['filename']
                space(data, member['size_bytes'])
                temporary = target_dir / (row['filename'] + '.' + uuid.uuid4().hex + '.part')
                digest = hashlib.sha256()
                written = 0
                with zipfile.ZipFile(data / member['archive']) as archive:
                    matches = [i for i in archive.infolist() if i.filename == member['member']
                               and i.file_size == member['size_bytes'] and i.CRC == int(member['crc32'], 16)]
                    if not matches:
                        raise ValueError('源 ZIP 成员在检查后发生变化')
                    with archive.open(matches[0]) as source, temporary.open('xb') as destination:
                        while block := source.read(CHUNK):
                            space(data, len(block))
                            destination.write(block)
                            digest.update(block)
                            written += len(block)
                        destination.flush()
                        os.fsync(destination.fileno())
                if written != member['size_bytes'] or digest.hexdigest() != member['sha256']:
                    raise ValueError(f'{target.name} 解压后 SHA256/大小不匹配；保留 .part')
                row.update(validate_hdf5(temporary))
                publish_no_replace(temporary, target)
                state['new_files'] += 1
                log(f'HDF5 PASS {kind}/{target.name} ({state["new_files"] + state["reused_files"]}/50)')
                snapshot()
            actual_ids = {int(EPISODE.fullmatch(p.name).group(1)) for p in target_dir.iterdir()
                          if EPISODE.fullmatch(p.name) and p.is_file() and not p.is_symlink()}
            if actual_ids != set(range(50)):
                raise ValueError('解压后编号检查失败')
            state['validated'] = True
            state['hdf5_count'] = len(actual_ids)
            paths = list(target_dir.glob('episode_*.hdf5'))
            state['logical_bytes'] = sum(p.stat().st_size for p in paths)
            state['allocated_bytes'] = sum(p.stat().st_blocks * 512 for p in paths)
        except Exception as error:
            state['errors'].append(f'{type(error).__name__}: {error}')
            if 'row' in locals():
                row.update(validation_result='验证失败', error=str(error))
            state['loading'] = '因数据校验失败未执行'
            log(f'{kind} STOP: {error}')
        snapshot()

    try:
        configuration = configure(root, states, reports)
    except Exception as error:
        configuration = '验证失败：配置未完成：' + str(error)
    atomic_text(reports / 'constants_configuration.txt', configuration + '\n')
    # Child process disables CUDA visibility, caps CPU parallelism and isolates errors.
    good = [k for k, s in states.items() if s['validated']]
    loading_output = reports / 'dataset_loading_test.txt'
    if good and not configuration.startswith(('尚未修改', '验证失败')):
        import subprocess
        result_file = reports / ('loading_result.' + uuid.uuid4().hex + '.json')
        command = [sys.executable, str(Path(__file__).resolve()), '--loading-only',
                   ','.join(good), '--loading-result', str(result_file)]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                           OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
        with loading_output.open('w', encoding='utf-8') as output:
            output.write(f'时间（上海）：{now()}\nPython: {sys.executable}\nCUDA_VISIBLE_DEVICES=\n')
            output.flush()
            try:
                completed = subprocess.run(command, cwd=root, env=environment, stdout=output,
                                           stderr=subprocess.STDOUT, timeout=180)
                if result_file.exists():
                    for kind, status in json.loads(result_file.read_text()).items():
                        states[kind]['loading'] = status
                else:
                    for kind in good:
                        states[kind]['loading'] = f'验证失败：官方加载子进程退出码 {completed.returncode}，详见日志'
            except Exception as error:
                output.write(traceback.format_exc())
                for kind in good:
                    states[kind]['loading'] = f'验证失败：{type(error).__name__}: {error}'
    else:
        atomic_text(loading_output, '\n'.join(f'{k}: {s["loading"]}' for k, s in states.items())
                    + '\n' + configuration + '\n')
    # Rehash source ZIPs at the end: original bytes and names must remain unchanged.
    preservation_errors = []
    for entry in inventory:
        path = Path(entry['path'])
        if not path.is_file() or path.stat().st_size != entry['size_bytes'] or sha256(path) != entry['sha256']:
            preservation_errors.append(entry['filename'])
    if sorted(p.name for p in data.iterdir() if p.suffix.lower() == '.zip') != [p.name for p in initial_zips]:
        preservation_errors.append('原始 ZIP 文件集合变化')
    snapshot()
    training_modules = ('einops', 'pyquaternion', 'mujoco', 'dm_control', 'cv2', 'rospkg')
    missing_training_modules = [name for name in training_modules
                                if importlib.util.find_spec(name) is None]
    training_note = ('- 验证解释器中检测不到的训练/仿真相关模块：'
                     + ('、'.join(f'`{name}`' for name in missing_training_modules) or '无（仅检查可发现性）')
                     + '。数据就绪不等于完整训练环境就绪；未安装依赖，应在独立且依赖齐全的 ACT 环境执行训练命令。')
    lines = ['# Transfer Cube 数据整理校验报告', '', f'- 时间（Asia/Shanghai）：{now()}',
             f'- ACT 项目绝对路径：`{root}`', f'- 数据绝对路径：`{data}`',
             f'- 执行解释器：`{sys.executable}`', f'- 环境：{versions}',
             '- 未安装或升级依赖；未修改 GPU/CUDA/PyTorch/MuJoCo；未启动模型训练。', training_note,
             f'- 初始可用空间：{initial_free} 字节；预检所需新增空间：{total_required} 字节，另保留 1 GiB。',
             f'- 原始 ZIP 共 {len(inventory)} 个，总计 {sum(z["size_bytes"] for z in inventory)} 字节。',
             f'- 原始 ZIP 保留验证：{"已成功验证：全部名称/大小/SHA256 与执行前一致" if not preservation_errors else "验证失败：" + str(preservation_errors)}',
             f'- constants.py：{configuration}', '',
             '| 数据集 | ZIP 数 | HDF5 数 | 逻辑大小 GB / GiB | 磁盘分配 GB | 格式校验 | 官方加载测试 |',
             '|---|---:|---:|---:|---:|---|---|']
    for kind, state in states.items():
        size = state.get('logical_bytes', 0)
        directory = data / ('sim_transfer_cube_' + kind)
        count = len(list(directory.glob('episode_*.hdf5'))) if directory.is_dir() else 0
        status = '已成功验证' if state['validated'] else ('验证失败' if state['errors'] else '因缺失数据无法执行')
        lines.append(f'| {kind} | {len(state["zips"])} | {count} | {size/1e9:.6f} / {size/1024**3:.6f} | '
                     f'{state.get("allocated_bytes",0)/1e9:.6f} | {status} | {state["loading"].splitlines()[0]} |')
    lines += ['', '## 已成功验证与验证方法', '',
              '- ZIP：独立结束记录/磁盘号检查；所有普通成员（含忽略的辅助文件）流式读取触发 zipfile CRC；保存 ZIP SHA256。',
              '- HDF5：按文件名白名单提取编号；比较所有同编号成员的完整解压内容 SHA256；原始目录见 archive_members.csv。',
              '- 无冲突的数据集才可解压；逐块检查磁盘空间，写入独占 .part 文件，校验后原子发布且禁止覆盖目标。',
              '- 每条轨迹必须有 sim=True/1；top=(400,480,640,3) uint8，qpos/qvel/action=(400,14) 实数。',
              '- 全部图像按 8 帧读取；全部 qpos/qvel/action 按 64 帧读取并检查 NaN/Inf；不修改 HDF5 内容。',
              '- 根据本地 utils.py：is_sim 真值走仿真对齐分支；图像按相机转 CHW 后 /255；状态/动作按官方统计量标准化。',
              '- 官方 load_data() 的 train/val 各读取一个 batch=2，并读取指定 episode_0；只用 CPU。实际结果见 dataset_loading_test.txt。', '']
    for kind, state in states.items():
        lines += [f'## {kind}', '']
        if not state['present']:
            lines += ['因缺失数据无法执行：未发现该数据集的压缩包或目录。', '']
            continue
        lines += [f'- 新提取 {state["new_files"]} 条，复用已有文件 {state["reused_files"]} 条。',
                  f'- 重复编号：{state.get("duplicate_episode_ids", [])}。',
                  f'- 官方加载状态：{state["loading"]}']
        if state['errors']:
            lines += ['- 验证失败，已停止该数据集后续整理：'] + ['  - ' + e for e in state['errors']]
        lines += ['- ' + w for w in state.get('warnings', [])] + ['']
    lines += ['## 尚未验证 / 不作结论', '',
              '- HDF5 格式完整仅证明可读取，未重放或统计任务成功率；不能据此声称专家成功率为 100%。',
              '- 未运行正式训练、MuJoCo 仿真或 GPU 可用性测试；训练环境的其他依赖未作全面验证。',
              '- 用户未给出 E1/E2/E3/E4 的定义，本仓库也未在上述训练配置中定义这些名称；只判断现有 Transfer Cube 数据条件，不推断其他任务条件。', '',
              '## 恢复步骤', '',
              '- ZIP 损坏/缺失：从原下载源补充正确压缩包，保留现有 ZIP，重新运行本脚本。',
              '- 同编号不同哈希：依据 archive_members.csv 核对下载来源并明确选用哪个完整版本；脚本不会自动选版本或覆盖。',
              '- 现有目标冲突/嵌套：先人工核对来源和哈希，另行授权处理冲突目标；脚本不会删除或覆盖。',
              '- 空间不足：增加可用空间后重新运行；已有完整 HDF5 将核验后复用，原 .part 保留供检查。',
              '- 依赖缺失：使用已有含 h5py/numpy 的解释器；官方加载另需 torch/IPython，详见加载日志。', '',
              '## 可重复执行', '', f'```bash\ncd {root}\n{sys.executable} prepare_transfer_cube_data.py\n```', '']
    if all(s['validated'] and s['loading'].startswith('已成功验证') for s in states.values()):
        lines += ['## 训练命令（仅提供，未执行）', '',
                  '以下使用仓库 README 的 ACT 参数；正式运行前应在已配置 ACT/MuJoCo 训练依赖的环境中执行。']
        for kind in ('scripted', 'human'):
            lines += ['', f'```bash\ncd {root}\npython imitate_episodes.py \\\n  --task_name sim_transfer_cube_{kind} \\\n  --ckpt_dir {root}/checkpoints/transfer_cube_{kind}_act_seed0 \\\n  --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 \\\n  --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0\n```']
    atomic_text(reports / 'validation_report.md', '\n'.join(lines) + '\n')
    log(f'FINISHED: report={reports / "validation_report.md"}')
    return 0 if not preservation_errors and all(
        not s['present'] or (s['validated'] and s['loading'].startswith('已成功验证')) for s in states.values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loading-only', default='', help=argparse.SUPPRESS)
    parser.add_argument('--loading-result', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.loading_only:
        try:
            result = loading_test(root, args.loading_only.split(','))
        except Exception:
            message = '验证失败：' + traceback.format_exc()
            print(message, flush=True)
            result = {k: message for k in args.loading_only.split(',')}
        atomic_text(args.loading_result, json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if all(v.startswith('已成功验证') for v in result.values()) else 1
    for filename in ('constants.py', 'utils.py', 'imitate_episodes.py', 'record_sim_episodes.py'):
        if not (root / filename).is_file():
            parser.error(f'ACT 项目文件缺失：{filename}')
    if not (root / 'data').is_dir() or (root / 'data').is_symlink():
        parser.error('data 必须是 ACT 项目内的真实目录')
    reports = root / 'data_preparation_reports'
    reports.mkdir(exist_ok=True)
    with (reports / '.preparation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('已有整理进程运行，拒绝并发修改')
        return run(root)


if __name__ == '__main__':
    sys.exit(main())
