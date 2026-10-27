"""Lightweight status command; does not load a model or render context."""
import argparse
import json
from pathlib import Path
import time
from common import RUN,MODELS,EVALS,TRAINING_SEEDS


def show():
    path=RUN/'reports/run_state.json'
    if not path.exists():
        print((RUN/'status.txt').read_text());return
    state=json.loads(path.read_text())
    pid=state.get('pid')
    alive=False
    if pid and Path('/proc/%d/cmdline'%pid).exists():
        command=Path('/proc/%d/cmdline'%pid).read_bytes().replace(b'\0',b' ').decode()
        alive=str(RUN/'scripts/pipeline.py') in command
    print('状态:',state['status'],'阶段:',state['phase'],'更新:',state['updated_at'])
    print('正式训练种子:',TRAINING_SEEDS,'模型:',len(MODELS),'评估组:',len(EVALS),'目标回合:',50*len(EVALS))
    print('调度器 PID:',pid,'实际存活:',alive,'训练并发:',state.get('train_parallel'),
          '评估并发:',state.get('eval_parallel'))
    print('训练期间评估并发:',state.get('evaluations_during_training',0))
    print('渲染: OSMesa LLVMpipe CPU；训练/推理: A800 CUDA GPU；NVIDIA EGL 未启用')
    for section,specs,progress,target in [('models',MODELS,'epochs_completed',2000),('evaluations',EVALS,'episodes_completed',50)]:
        for spec in specs:
            rec=state[section][spec['name']];data={}
            if rec.get('directory'):
                path=Path(rec['directory'])/'progress.json'
                if path.exists():data=json.loads(path.read_text())
            print('%-30s %-14s %d/%d'%(spec['name'],rec['status'],data.get(progress,0),target))
    print('finish.txt:',(RUN/'finish.txt').exists())
    if state.get('error'):print('错误:',state['error'])
    if state['status']=='RUNNING' and not alive:
        print('注意：状态文件仍写 RUNNING，但调度器已退出。检查日志并使用 --continue_run；不要直接复用部分训练目录。')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        show()
        if not args.watch:break
        time.sleep(15)
