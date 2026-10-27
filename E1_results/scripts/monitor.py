"""Read-only monitor; runs with system Python as well as act-repro."""
import json
import os
from pathlib import Path
import re
import subprocess
from common import RUN, MODELS

def alive(pid):
    try: os.kill(pid,0); return True
    except ProcessLookupError: return False

def main():
    path=RUN/'reports/run_state.json'
    if not path.exists():
        print('No scheduler state yet. Inspect preflight logs.'); return
    state=json.loads(path.read_text())
    print('Run:',RUN)
    print('Status:',state['status'],'phase:',state['phase'])
    print('Scheduler PID:',state['pid'],'alive:',alive(state['pid']))
    for name,policy,chunk in MODELS:
        model=state['models'].get(name,{'status':'PENDING'})
        extra=''
        if 'directory' in model:
            log=Path(model['directory'])/'train.stdout.log'
            if log.exists():
                epochs=re.findall(r'^Epoch (\d+)$',log.read_text(errors='replace'),re.M)
                extra='; current epoch='+str(int(epochs[-1])+1)+'/1000' if epochs else '; initializing'
        evaluation=state['evaluations'].get(name,{'status':'PENDING'})
        eval_extra=''
        if 'directory' in evaluation:
            log=Path(evaluation['directory'])/'eval.stdout.log'
            if log.exists(): eval_extra='; completed rollouts='+str(len(re.findall(r'^Rollout \d+$',log.read_text(errors='replace'),re.M)))+'/30'
        print(name,'training:',model['status']+extra,'evaluation:',evaluation['status']+eval_extra)
    for name,test in state.get('tests',{}).items(): print('Check:',name,test['status'])
    if 'formal_training_wall_seconds' in state: print('Formal training wall seconds:',state['formal_training_wall_seconds'])
    if 'error' in state: print('ERROR:',state['error'])
    print('finish.txt exists:',(RUN/'finish.txt').exists())
    try: subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv'],check=False)
    except FileNotFoundError: pass

if __name__=='__main__': main()
