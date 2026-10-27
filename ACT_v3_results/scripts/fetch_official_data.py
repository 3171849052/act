"""System-Python public downloads into missing task directories; never overwrite."""
import concurrent.futures
import fcntl
import json
import os
import subprocess
from pathlib import Path,PurePosixPath
import sys
import traceback
from common import RUN,TASKS,PYTHON,now,write_json
sys.path.insert(0,str(RUN/'runtime/vendor'))


def main():
    import gdown
    lock=(RUN/'downloads/fetch.lock').open('a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    inventory=json.loads((RUN/'provenance/official_data_inventory.json').read_text())
    def task_download(spec):
        data=Path(spec['dataset_dir']);data.mkdir(parents=True,exist_ok=True)
        temp=RUN/'downloads'/spec['task_name'];temp.mkdir(parents=True,exist_ok=True)
        records=[r for r in inventory if len(PurePosixPath(r['path']).parts)==2 and
            PurePosixPath(r['path']).parts[0]==spec['task_name'] and r['path'].endswith('.hdf5')]
        assert {PurePosixPath(r['path']).name for r in records}=={'episode_%d.hdf5'%i for i in range(50)}
        state=dict(task_name=spec['task_name'],status='DOWNLOADING',completed=0,started_at=now(),
            source='Public dataset folder linked by official ACT README',files=[])
        report=RUN/'reports/downloads'/(spec['name']+'.json')
        write_json(report,state)
        try:
            for record in sorted(records,key=lambda r:int(PurePosixPath(r['path']).stem.split('_')[-1])):
                filename=PurePosixPath(record['path']).name;target=data/filename
                if target.exists():
                    state['completed']+=1;continue
                path=temp/(filename+'.partial')
                gdown.download(id=record['id'],output=str(path),quiet=True,use_cookies=False,resume=True,
                    speed=20*1024**2,timeout=60,retries=2)
                with path.open('rb') as f:assert f.read(8)==b'\x89HDF\r\n\x1a\n'
                metadata_check="import sys,h5py; f=h5py.File(sys.argv[1],'r'); n=int(sys.argv[2]); assert f['action'].shape==(n,14); assert f['observations/images/top'].shape==(n,480,640,3); assert bool(f.attrs['sim']); f.close()"
                subprocess.run([PYTHON,'-c',metadata_check,str(path),str(spec['episode_len'])],check=True,
                               env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1'))
                try:os.link(str(path),str(target))
                except FileExistsError:pass
                else:path.unlink()
                state['completed']+=1;state['updated_at']=now()
                state['files'].append(dict(path=str(target),google_drive_id=record['id'],bytes=target.stat().st_size))
                write_json(report,state)
                print('Downloaded',spec['task_name'],filename,flush=True)
            state.update(status='DOWNLOADED',completed_at=now())
        except Exception:
            state.update(status='BLOCKED',error=traceback.format_exc(),updated_at=now())
            traceback.print_exc()
        write_json(report,state)
        return state
    needed=[s for s in TASKS if len(list(Path(s['dataset_dir']).glob('episode_*.hdf5')))!=50]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        result=list(executor.map(task_download,needed))
    write_json(RUN/'reports/downloads/final.json',dict(tasks=result,completed_at=now()))
    return 0 if all(r['status']=='DOWNLOADED' for r in result) else 1


if __name__=='__main__':sys.exit(main())
