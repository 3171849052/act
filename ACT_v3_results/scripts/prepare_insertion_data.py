"""Prepare local Insertion ZIPs without overwriting data or enabling training.

All ZIP members are streamed through CRC checks. Each unique HDF5 is staged,
fully validated and hashed before no-replace publication to its own dataset.
"""
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path,PurePosixPath
import pickle
import re
import shutil
import stat
import sys
import time
import traceback
import uuid
import zipfile
from common import ROOT,RUN,ALL_TASKS,now,write_json,sha256
sys.path.insert(0,str(ROOT))
from prepare_transfer_cube_data import check_single_disk,publish_no_replace

OUT=RUN/'reports/insertion_data_preparation'
EPISODE=re.compile(r'episode_(0|[1-9][0-9]*)\.hdf5\Z')
BLOCK=8*1024*1024


def save_csv(path,rows,fields):
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=fields,extrasaction='ignore')
    writer.writeheader();writer.writerows(rows)
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(buffer.getvalue());os.replace(str(temporary),str(path))


def validate_hdf5(path,length):
    import h5py
    import numpy as np
    result={}
    with h5py.File(str(path),'r') as f:
        sim=np.asarray(f.attrs['sim'])
        assert sim.shape==() and sim.dtype.kind in 'biu' and sim.item()==1,'Invalid simulation flag'
        result['sim']=bool(sim.item())
        shapes={'action':(length,14),'observations/qpos':(length,14),'observations/qvel':(length,14),
                'observations/images/top':(length,480,640,3)}
        for name,shape in shapes.items():
            node=f
            for component in name.split('/'):
                assert isinstance(node.get(component,getlink=True),h5py.HardLink),'External/soft HDF5 link: '+name
                node=node[component]
            assert isinstance(node,h5py.Dataset) and not node.is_virtual and not node.external
            assert node.shape==shape,(name,node.shape,shape)
            image=name.endswith('/top')
            assert node.dtype==np.uint8 if image else node.dtype.kind in 'fiu'
            low=float('inf');high=float('-inf')
            for start in range(0,length,8 if image else 64):
                data=node[start:start+(8 if image else 64)]
                assert np.isfinite(data).all(),name+' contains NaN/Inf'
                low=min(low,float(data.min()));high=max(high,float(data.max()))
            if image:assert low<high,'Image stream is constant'
            else:assert max(abs(low),abs(high))<1e6,'Implausible trajectory values'
            result[name]=dict(shape=list(shape),dtype=str(node.dtype),min=low,max=high)
    return result


def loader_check(spec,files):
    import numpy as np
    import torch
    import utils
    from constants import SIM_TASK_CONFIGS
    torch.set_num_threads(1)
    config=SIM_TASK_CONFIGS[spec['task_name']]
    assert Path(config['dataset_dir']).resolve()==Path(spec['dataset_dir']).resolve()
    assert config['episode_len']==spec['episode_len'] and config['num_episodes']==50 and config['camera_names']==['top']
    utils.set_seed(1)
    train,val,stats,is_sim=utils.load_data(spec['dataset_dir'],50,['top'],8,8)
    assert is_sim
    split=dict(train=train.dataset.episode_ids.tolist(),validation=val.dataset.episode_ids.tolist())
    assert sorted(split['train']+split['validation'])==list(range(50))
    assert len(split['train'])==40 and len(split['validation'])==10
    shapes={}
    for label,loader in [('train',train),('validation',val)]:
        data=next(iter(loader));actual=[list(t.shape) for t in data]
        assert actual==[[8,1,3,480,640],[8,14],[8,spec['episode_len'],14],[8,spec['episode_len']]]
        assert all(t.device.type=='cpu' and torch.isfinite(t).all().item() for t in data)
        assert data[0].dtype==torch.float32 and data[3].dtype==torch.bool
        assert 0<=data[0].min().item()<=data[0].max().item()<=1
        shapes[label]=actual
    directory=RUN/'reports/data'/spec['name'];directory.mkdir(parents=True,exist_ok=True)
    temporary=directory/'reference_stats.pkl.tmp'
    with temporary.open('wb') as f:pickle.dump(stats,f)
    os.replace(str(temporary),str(directory/'reference_stats.pkl'))
    report=dict(passed=True,task=spec,files=files,split_seed=1,split=split,batch_shapes=shapes,
                completed_at=now(),normalization_scope='Official all-50 statistics for this dataset only',
                sim_flag_meaning='Simulation flag, not a scripted/human provenance label',
                source='Local ZIP extraction with complete CRC, SHA256 and all-frame validation')
    write_json(directory/'validation.json',report)
    return dict(passed=True,split=split,batch_shapes=shapes,python=sys.executable,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                normalization_source=spec['dataset_dir'],stats_sha256=sha256(directory/'reference_stats.pkl'))


def prepare(spec,members,archives,stage):
    name=spec['task_name'];target=Path(spec['dataset_dir'])
    assert not target.is_symlink();target.mkdir(parents=True,exist_ok=True)
    existing={}
    for path in target.iterdir():
        if path.suffix.lower() not in ['.hdf5','.h5']:continue
        match=EPISODE.fullmatch(path.name)
        assert match and path.is_file() and not path.is_symlink(),'Unexpected existing dataset entry: '+str(path)
        existing[int(match[1])]=path
    assert not set(existing)-set(range(50))
    local=stage/name;local.mkdir()
    seen={};sources={};archive_rows=[]
    for archive in archives:
        path=ROOT/'data'/archive['filename']
        assert path.stat().st_size==archive['bytes'] and path.stat().st_mtime_ns==archive['mtime_ns'],'Archive changed during preparation'
        check_single_disk(path)
        with zipfile.ZipFile(str(path)) as z:
            for info in z.infolist():
                parts=PurePosixPath(info.filename).parts
                assert not info.filename.startswith('/') and '..' not in parts and '\\' not in info.filename
                mode=info.external_attr>>16
                assert not stat.S_ISLNK(mode) and stat.S_IFMT(mode) in [0,stat.S_IFREG,stat.S_IFDIR]
                assert not info.flag_bits&1,'Encrypted archive member'
                if info.is_dir():continue
                base=PurePosixPath(info.filename).name;match=EPISODE.fullmatch(base)
                if base.lower().endswith(('.hdf5','.h5')):
                    assert match,'Unexpected HDF5 filename: '+info.filename
                    assert all(not p.startswith('sim_') or p==name for p in parts[:-1]),'Dataset type mismatch'
                    episode=int(match[1]);assert 0<=episode<50
                else:episode=None
                destination=None;temporary=None;digest=hashlib.sha256();written=0
                if episode is not None and episode not in seen and episode not in existing:
                    temporary=local/(base+'.partial');destination=temporary.open('xb')
                try:
                    with z.open(info) as src:
                        while True:
                            block=src.read(BLOCK)
                            if not block:break
                            digest.update(block);written+=len(block)
                            if destination:destination.write(block)
                    if destination:destination.flush();os.fsync(destination.fileno())
                finally:
                    if destination:destination.close()
                assert written==info.file_size
                checksum=digest.hexdigest()
                archive_rows.append(dict(archive=path.name,member=info.filename,episode=episode if episode is not None else '',
                    bytes=written,crc32='%08x'%info.CRC,sha256=checksum))
                if episode is None:continue
                sources.setdefault(episode,[]).append(dict(archive=path.name,member=info.filename))
                if episode in seen:
                    assert seen[episode]['sha256']==checksum,'Conflicting duplicate episode: '+base
                    continue
                validation_path=existing.get(episode,temporary)
                if episode in existing:assert sha256(validation_path)==checksum,'Existing dataset differs from archive; refuse overwrite: '+base
                details=validate_hdf5(validation_path,spec['episode_len'])
                seen[episode]=dict(id=episode,path=str(target/base),bytes=written,sha256=checksum,validation=details,
                                   staged=str(temporary) if temporary else None,reused=episode in existing)
                print(now(),name,base,'CRC / SHA256 / full HDF5 PASS',flush=True)
                write_json(OUT/'progress.json',dict(status='VALIDATING',task=name,validated=len(seen),expected=50,time=now()))
        assert path.stat().st_size==archive['bytes'] and path.stat().st_mtime_ns==archive['mtime_ns']
    assert set(seen)|set(existing)==set(range(50)),'Incomplete episode coverage'
    for episode,path in existing.items():
        if episode not in seen:
            seen[episode]=dict(id=episode,path=str(path),bytes=path.stat().st_size,sha256=sha256(path),
                validation=validate_hdf5(path,spec['episode_len']),staged=None,reused=True)
    # No user-visible publication until every archive/member/duplicate of this dataset passed.
    for episode,record in sorted(seen.items()):
        if record['staged']:
            source=Path(record['staged']);destination=Path(record['path'])
            try:publish_no_replace(source,destination)
            except FileExistsError:assert sha256(destination)==record['sha256'],'Concurrent target conflict; refuse overwrite'
        record.pop('staged',None);record['sources']=sources.get(episode,[])
    assert {p.name for p in target.glob('episode_*.hdf5')}=={'episode_%d.hdf5'%i for i in range(50)}
    files=[seen[i] for i in range(50)];loading=loader_check(spec,files)
    members.extend(archive_rows)
    return dict(ready=True,task_name=name,episode_len=spec['episode_len'],files=files,official_loader=loading,
                new_files=sum(not r['reused'] for r in files),reused_files=sum(r['reused'] for r in files),archives=len(archives))


def main():
    os.nice(15);OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'preparation.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    specs=[s for s in ALL_TASKS if s['task_name'].startswith('sim_insertion_')]
    stamp=now();inventory={};expected_bytes=0
    for spec in specs:
        archives=[];ids=set()
        for path in sorted((ROOT/'data').glob(spec['task_name']+'*.zip')):
            before=path.stat();check_single_disk(path)
            entry=dict(filename=path.name,bytes=before.st_size,mtime_ns=before.st_mtime_ns,sha256=sha256(path),members=[])
            with zipfile.ZipFile(str(path)) as z:
                for info in z.infolist():
                    if not info.filename.lower().endswith(('.hdf5','.h5')):continue
                    match=EPISODE.fullmatch(PurePosixPath(info.filename).name);assert match
                    episode=int(match[1]);ids.add(episode);expected_bytes+=info.file_size
                    entry['members'].append(dict(episode=episode,name=info.filename,bytes=info.file_size))
            assert path.stat().st_size==before.st_size and path.stat().st_mtime_ns==before.st_mtime_ns
            archives.append(entry)
        assert ids==set(range(50)),spec['task_name']+' missing: '+str(sorted(set(range(50))-ids))
        inventory[spec['task_name']]=archives
    write_json(OUT/'archive_inventory.json',inventory)
    assert shutil.disk_usage(str(ROOT/'data')).free>expected_bytes+2*1024**3
    stage=ROOT/'data'/('.insertion_staging_'+uuid.uuid4().hex);stage.mkdir()
    write_json(OUT/'preparation_results.json',dict(status='RUNNING',started_at=stamp,stage=str(stage)))
    results={};members=[]
    for spec in specs:
        try:results[spec['task_name']]=prepare(spec,members,inventory[spec['task_name']],stage)
        except Exception:
            results[spec['task_name']]=dict(ready=False,error=traceback.format_exc());traceback.print_exc()
        write_json(OUT/'preparation_results.json',dict(status='RUNNING',started_at=stamp,stage=str(stage),datasets=results))
    unchanged=all(sha256(ROOT/'data'/a['filename'])==a['sha256'] and (ROOT/'data'/a['filename']).stat().st_size==a['bytes']
                  for archives in inventory.values() for a in archives)
    done=all(r['ready'] for r in results.values()) and unchanged
    report=dict(status='COMPLETED' if done else 'BLOCKED',started_at=stamp,completed_at=now(),datasets=results,
                archive_bytes_preserved=unchanged,training_automatically_enabled=False,stage=str(stage))
    write_json(OUT/'preparation_results.json',report)
    save_csv(OUT/'archive_members.csv',members,['archive','member','episode','bytes','crc32','sha256'])
    flat=[dict(task_name=task,id=f['id'],path=f['path'],bytes=f['bytes'],sha256=f['sha256'],episode_len=r['episode_len'])
          for task,r in results.items() if r['ready'] for f in r['files']]
    save_csv(OUT/'dataset_manifest.csv',flat,['task_name','id','path','bytes','sha256','episode_len'])
    lines=['# Insertion 数据整理与校验','', '时间：'+report['completed_at'],'','| 数据集 | 文件数 | 轨迹长度 | 原始 ZIP | 官方加载器 | 就绪 |',
           '| --- | ---: | ---: | ---: | --- | --- |']
    for spec in specs:
        r=results[spec['task_name']]
        lines.append('| %s | %d | %d | %d | %s | %s |'%(spec['task_name'],len(r.get('files',[])),spec['episode_len'],
            len(inventory[spec['task_name']]),'通过：train/validation 各一个 batch=8' if r['ready'] else '失败/未执行',r['ready']))
        if not r['ready']:lines+=['','```',r['error'],'```']
    lines+=['','逐个普通 ZIP 和全部成员流式 CRC 校验；校验重复编号 SHA256，完整检查每条 HDF5 的所有图像、qpos/qvel/action、NaN/Inf、sim 标志和字段形状。',
        'Human 使用 500 步，Scripted 使用 400 步，均为 top RGB (480,640,3) uint8；动作与状态每步 14 维。',
        '官方 utils.load_data 实际读取独立数据集，固定划分种子 1，40 条训练 / 10 条验证；CPU 执行，CUDA_VISIBLE_DEVICES 为空。分别生成各自归一化统计，不跨数据集使用。',
        '原始 ZIP 名称、大小及 SHA256 保留验证：'+str(unchanged)+'。没有覆盖任何已有 HDF5，没有修改 constants.py、V1/V2 结果或正在运行的 Cube 任务。',
        '数据校验就绪不等于专家成功率已验证；未做专家回放，不声称示范成功率 100%。',
        '本次只整理数据；Insertion 训练开关仍保持关闭，未自动启动新训练或仿真评估。',
        '详细清单：archive_inventory.json、archive_members.csv、dataset_manifest.csv、preparation_results.json。']
    (OUT/'data_preparation_report.md').write_text('\n'.join(lines)+'\n')
    (OUT/'status.txt').write_text(report['status']+'\n'+report['completed_at']+'\n')
    print('INSERTION DATA PREPARATION',report['status'],flush=True)
    return 0 if done else 1


if __name__=='__main__':sys.exit(main())
