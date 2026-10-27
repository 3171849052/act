"""V3 supervisor: low-priority CUDA training overlaps V2; evaluation stays gated."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from common import ROOT,RUN,PYTHON,TASKS,ALL_TASKS,PLAN,now,write_json,worker_environment,validate_model
from v2_dependency import inspect_v2

STATE={};ACTIVE={};DATA_JOBS={};LAST_MONITOR=0.


def save(status=None):
    if status:STATE['status']=status
    STATE['updated_at']=now();write_json(RUN/'reports/run_state.json',STATE)
    lines=['ACT_V3 '+STATE['status'],'更新时间: '+STATE['updated_at'],'调度器 PID: '+str(os.getpid()),
        'CUDA 训练；OSMesa LLVMpipe CPU 仿真；seed=0；V3 正式评估等待 V2 完成',
        '当前训练并发上限: '+str(STATE.get('train_parallel',1)),
        'V2: '+STATE.get('v2',{}).get('condition','checking')]
    for group in ['models','evaluations']:
        for spec in TASKS:
            rec=STATE[group][spec['name']];progress=''
            if rec.get('directory'):
                path=Path(rec['directory'])/'progress.json'
                if path.exists():
                    data=json.loads(path.read_text());rec['progress']=data
                    key='epochs_completed' if group=='models' else 'episodes_completed'
                    progress=' %d/%d'%(data.get(key,0),2000 if group=='models' else 50)
            lines.append(group+' '+spec['name']+': '+rec['status']+progress)
    if STATE.get('error'):lines.append('错误: '+STATE['error'])
    lines.append('finish.txt: '+str((RUN/'finish.txt').exists()))
    lines += ['暂停 '+s['name']+': 用户准备数据，尚未启用' for s in ALL_TASKS if not s.get('enabled',True)]
    temp=RUN/'status.txt.tmp';temp.write_text('\n'.join(lines)+'\n');os.replace(str(temp),str(RUN/'status.txt'))
    write_json(RUN/'reports/run_state.json',STATE)


def monitor():
    global LAST_MONITOR
    if time.monotonic()-LAST_MONITOR<30:return
    LAST_MONITOR=time.monotonic()
    row=dict(time=now(),phase=STATE['status'],cpu_stat=Path('/sys/fs/cgroup/cpu.stat').read_text(),
        cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        memory_bytes=int(Path('/sys/fs/cgroup/memory.current').read_text()),workers={name:p.pid for name,(p,_) in ACTIVE.items()})
    try:row['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total',
        '--format=csv,noheader'],text=True,timeout=10).strip()
    except Exception as e:row['gpu_query_error']=repr(e)
    with (RUN/'logs/resource_monitor.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')


def fresh(path):
    if not path.exists():path.mkdir(parents=True);return path
    n=1
    while path.with_name(path.name+'_attempt%d'%n).exists():n+=1
    path=path.with_name(path.name+'_attempt%d'%n);path.mkdir(parents=True);return path


def start_worker(spec,phase,directory,episodes=None):
    if phase=='eval':
        model=Path(STATE['models'][spec['name']]['directory'])
        for name in ['policy_best.ckpt','dataset_stats.pkl']:(directory/name).symlink_to(model/name)
    cmd=['nice','-n','10',PYTHON,'-u',str(RUN/'scripts/worker.py'),'--phase',phase,'--task',spec['name'],
         '--output',str(directory),'--epochs','2000']
    if episodes:cmd+=['--smoke_episodes',str(episodes)]
    with (directory/'stdout.log').open('x') as out,(directory/'stderr.log').open('x') as err:
        p=subprocess.Popen(cmd,cwd=str(ROOT),env=worker_environment(),stdout=out,stderr=err,start_new_session=True,stdin=subprocess.DEVNULL)
    rec=dict(status='RUNNING',pid=p.pid,command=cmd,directory=str(directory),phase=phase,started_at=now(),start_monotonic=time.monotonic())
    write_json(directory/'job.json',rec);return p,rec


def data_ready(spec):
    report=RUN/'reports/data'/spec['name']/'validation.json'
    if report.exists() and json.loads(report.read_text()).get('passed'):return True
    if spec['name'] in DATA_JOBS:
        p=DATA_JOBS[spec['name']]
        if p.poll() is None:return False
        del DATA_JOBS[spec['name']]
        if p.returncode:
            STATE['models'][spec['name']].update(status='BLOCKED_DATA',reason='Data validation failed; see task data logs')
            return False
        return report.exists() and json.loads(report.read_text()).get('passed')
    if STATE['models'][spec['name']]['status']=='BLOCKED_DATA':return False
    directory=Path(spec['dataset_dir'])
    if len(list(directory.glob('episode_*.hdf5')))!=50:
        download=RUN/'reports/downloads'/(spec['name']+'.json')
        if download.exists() and json.loads(download.read_text()).get('status')=='BLOCKED':
            STATE['models'][spec['name']].update(status='BLOCKED_DATA',reason='Official data download blocked; source evidence and error retained')
        return False
    # Only one full dataset scan at once; training/evaluation retain CPU priority.
    if DATA_JOBS:return False
    out=RUN/'logs'/('data_'+spec['name']+'.stdout.log');err=RUN/'logs'/('data_'+spec['name']+'.stderr.log')
    with out.open('a') as stdout,err.open('a') as stderr:
        p=subprocess.Popen(['ionice','-c','3',PYTHON,'-u',str(RUN/'scripts/preflight.py'),'--task',spec['name']],
            cwd=str(ROOT),env=worker_environment(),stdout=stdout,stderr=stderr,start_new_session=True,stdin=subprocess.DEVNULL)
    DATA_JOBS[spec['name']]=p;return False


def finish_worker(name,p,rec,group):
    rec.update(status='SUCCEEDED' if p.returncode==0 else 'FAILED',exit_code=p.returncode,ended_at=now(),
               elapsed_seconds=time.monotonic()-rec['start_monotonic'])
    write_json(Path(rec['directory'])/'job.json',rec)
    STATE[group][name]['status']=rec['status']
    if p.returncode:raise RuntimeError(name+' '+group+' failed; logs preserved at '+rec['directory'])
    marker='artifact_validation.json' if group=='models' else 'evaluation_metrics.json'
    STATE[group][name]['artifacts' if group=='models' else 'metrics']=json.loads((Path(rec['directory'])/marker).read_text())


def train():
    while not all(STATE['models'][s['name']]['status']=='SUCCEEDED' for s in TASKS):
        dependency=inspect_v2();STATE['v2']=dependency
        cap=PLAN['execution']['after_v2_training_processes'] if dependency['completed'] else PLAN['execution']['overlap_training_processes']
        cap=min(cap,STATE.get('train_cap_override',3));STATE['train_parallel']=cap
        permitted=dependency['completed'] or (PLAN['execution']['training_overlap_with_v2_authorized'] and dependency['training_overlap_allowed'])
        for spec in TASKS:
            rec=STATE['models'][spec['name']]
            if rec['status'] in ['SUCCEEDED','RUNNING']:continue
            ready=data_ready(spec)
            if ready and permitted and len(ACTIVE)<cap:
                directory=fresh(RUN/'models'/spec['name']);p,attempt=start_worker(spec,'train',directory)
                rec.setdefault('attempts',[]).append(attempt);rec.update(status='RUNNING',directory=str(directory),pid=p.pid)
                ACTIVE[spec['name']]=(p,attempt);save('TRAINING')
        for name,(p,rec) in list(ACTIVE.items()):
            if p.poll() is None:continue
            del ACTIVE[name]
            if p.returncode and 'out of memory' in (Path(rec['directory'])/'stderr.log').read_text(errors='replace').lower() and cap>1:
                rec.update(status='FAILED_OOM',exit_code=p.returncode,ended_at=now());write_json(Path(rec['directory'])/'job.json',rec)
                STATE['train_cap_override']=cap-1;STATE['models'][name]['status']='RETRY_PENDING'
            else:finish_worker(name,p,rec,'models')
        if ACTIVE:status='TRAINING'
        elif dependency['condition']=='BLOCKED_BY_V2':status='BLOCKED_BY_V2'
        elif any(r['status']=='BLOCKED_DATA' for r in STATE['models'].values()):status='BLOCKED_MISSING_DATA'
        else:status='DATA_VALIDATION'
        save(status);monitor();time.sleep(10)


def wait_for_evaluation():
    while True:
        dependency=inspect_v2();STATE['v2']=dependency
        plan=json.loads((RUN/'experiment_plan.json').read_text())
        if dependency['completed'] and plan['execution'].get('evaluation_implementation_ready',False):
            STATE['v2_completed_verified_at']=now();write_json(RUN/'reports/v2_completion_evidence.json',dependency);save('EVALUATING');return
        save('BLOCKED_BY_V2' if dependency['condition']=='BLOCKED_BY_V2' else 'QUEUED_WAITING_FOR_V2')
        monitor();time.sleep(60)


def run_command(script,args=()):
    with (RUN/'logs'/(Path(script).stem+'.stdout.log')).open('a') as out,(RUN/'logs'/(Path(script).stem+'.stderr.log')).open('a') as err:
        subprocess.run(['nice','-n','10',PYTHON,'-u',str(RUN/'scripts'/script)]+list(args),cwd=str(ROOT),
                       env=worker_environment(),stdout=out,stderr=err,check=True)


def evaluate():
    run_command('validate_render.py')
    pending=[s for s in TASKS if STATE['evaluations'][s['name']]['status']!='SUCCEEDED']
    results=[]
    for count in ([1,2,3] if len(pending)>1 else []):
        jobs=[];begin=time.monotonic()
        for i in range(count):
            directory=fresh(RUN/'tests'/('eval_parallel_%d_%d'%(count,i)))
            jobs.append(start_worker(TASKS[0],'eval',directory,episodes=1))
        while any(p.poll() is None for p,_ in jobs):monitor();time.sleep(5)
        wall=time.monotonic()-begin
        stable=all(p.returncode==0 for p,_ in jobs)
        results.append(dict(concurrency=count,stable=stable,wall_seconds=wall,episodes=count,
                            episodes_per_second=count/wall if stable else None,directories=[r['directory'] for _,r in jobs]))
    good=[r for r in results if r['stable']]
    assert good or len(pending)<=1,'All evaluation concurrency tests failed'
    cap=max(good,key=lambda r:r['episodes_per_second'])['concurrency'] if good else 1
    STATE['eval_parallel']=cap
    write_json(RUN/'reports/evaluation_throughput.json',dict(results=results,selected_concurrency=cap,
        reason='Only one active evaluation: run it once at 50 episodes, no extra throughput rollouts' if len(pending)<=1 else 'Measured concurrency'))
    queue=[s for s in TASKS if STATE['evaluations'][s['name']]['status']!='SUCCEEDED']
    while queue or ACTIVE:
        while queue and len(ACTIVE)<cap:
            spec=queue.pop(0);directory=fresh(RUN/'evaluations'/spec['name']);p,attempt=start_worker(spec,'eval',directory)
            rec=STATE['evaluations'][spec['name']];rec.setdefault('attempts',[]).append(attempt)
            rec.update(status='RUNNING',directory=str(directory),pid=p.pid);ACTIVE[spec['name']]=(p,attempt)
        for name,(p,rec) in list(ACTIVE.items()):
            if p.poll() is None:continue
            del ACTIVE[name];finish_worker(name,p,rec,'evaluations')
        save('EVALUATING');monitor();time.sleep(10)


def main():
    global STATE
    parser=argparse.ArgumentParser();parser.add_argument('--continue_run',action='store_true');a=parser.parse_args()
    lock=(RUN/'pipeline.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
    if a.continue_run:
        STATE=json.loads((RUN/'reports/run_state.json').read_text())
        for group in ['models','evaluations']:
            for name,rec in STATE[group].items():
                if rec['status']=='RUNNING':
                    path=Path('/proc/%d/cmdline'%rec['pid'])
                    assert not path.exists() or str(RUN/'scripts/worker.py').encode() not in path.read_bytes(),'Prior worker alive; do not duplicate'
                    marker=Path(rec['directory'])/'worker_completion.json'
                    if marker.exists():
                        rec['status']='SUCCEEDED';file='artifact_validation.json' if group=='models' else 'evaluation_metrics.json'
                        rec['artifacts' if group=='models' else 'metrics']=json.loads((Path(rec['directory'])/file).read_text())
                    else:rec['status']='INTERRUPTED'
        STATE.pop('error',None)
    else:
        assert not (RUN/'reports/run_state.json').exists()
        STATE=dict(status='STARTING',started_at=now(),models={s['name']:dict(status='PENDING') for s in TASKS},
                   evaluations={s['name']:dict(status='PENDING') for s in TASKS})
    STATE['pid']=os.getpid();(RUN/'pipeline.pid').write_text(str(os.getpid())+'\n')
    try:
        while True:
            save('DATA_VALIDATION');train();wait_for_evaluation();evaluate();save('REPORTING')
            run_command('collect_results.py',['--partial'])
            if all(STATE['models'][s['name']]['status']=='SUCCEEDED' and STATE['evaluations'][s['name']]['status']=='SUCCEEDED' for s in ALL_TASKS):
                run_command('collect_results.py');save('COMPLETED');break
            while True:
                save('WAITING_FOR_USER_DATA');monitor();time.sleep(60)
                new_plan=json.loads((RUN/'experiment_plan.json').read_text())
                enabled=[s for s in new_plan['tasks'] if s.get('enabled',True)]
                if {s['name'] for s in enabled}!={s['name'] for s in TASKS}:
                    PLAN.clear();PLAN.update(new_plan);TASKS[:]=enabled;ALL_TASKS[:]=new_plan['tasks']
                    for s in TASKS:
                        for group in ['models','evaluations']:
                            if STATE[group][s['name']]['status']=='WAITING_FOR_USER_DATA':STATE[group][s['name']]['status']='PENDING'
                    break
    except BaseException as exc:
        STATE.update(error=repr(exc),traceback=traceback.format_exc())
        for name,(p,rec) in ACTIVE.items():
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM);STATE['models' if rec['phase']=='train' else 'evaluations'][name]['status']='INTERRUPTED'
        save('FAILED');(RUN/'reports/failure_report.md').write_text(STATE['traceback']+'\n未创建完成标志；部分目录保留，权重不是完整优化器恢复点。\n')
        raise


if __name__=='__main__':main()
