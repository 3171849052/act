"""Read-only lightweight monitor; no model, MuJoCo or CUDA imports."""
import argparse
import json
from pathlib import Path
import time
from common import RUN,ALL_TASKS


def show():
    path=RUN/'reports/run_state.json'
    if not path.exists():print((RUN/'status.txt').read_text());return
    state=json.loads(path.read_text());pid=state.get('pid');p=Path('/proc/%s/cmdline'%pid)
    alive=p.exists() and str(RUN/'scripts/pipeline.py').encode() in p.read_bytes().split(b'\0')
    print('V3:',state['status'],'更新:',state['updated_at'],'监督 PID:',pid,'存活:',alive)
    print('训练使用 CUDA；仿真使用 CPU OSMesa；评估需 V2 验收完成。')
    for spec in ALL_TASKS:
        for group in ['models','evaluations']:
            rec=state[group][spec['name']];progress={}
            if rec.get('directory'):
                p=Path(rec['directory'])/'progress.json'
                if p.exists():progress=json.loads(p.read_text())
            key='epochs_completed' if group=='models' else 'episodes_completed';target=2000 if group=='models' else 50
            print(group,spec['name'],rec['status'],'%d/%d'%(progress.get(key,0),target))
    print('finish.txt:',(RUN/'finish.txt').exists())
    if state.get('error'):print('错误:',state['error'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');a=parser.parse_args()
    while True:
        show()
        if not a.watch:break
        time.sleep(15)
