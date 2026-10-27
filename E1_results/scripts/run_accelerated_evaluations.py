"""Migrate owned evaluation jobs to three independent, camera-verified workers."""
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from common import RUN,ROOT,MODELS,write_json,now,parse_evaluation,sha256

STATE=None
ACTIVE={}

def alive(pid):
    try:
        stat=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]
        return stat!='Z'
    except FileNotFoundError: return False

def stop_owned(pid,required_script):
    if not alive(pid): return
    proc=Path('/proc')/str(pid)
    assert proc.stat().st_uid==os.getuid()
    command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert required_script in command and str(RUN) in command,(pid,command)
    os.kill(pid,signal.SIGTERM)
    deadline=time.time()+25
    while alive(pid) and time.time()<deadline: time.sleep(0.25)
    assert not alive(pid), 'Owned process did not stop; refusing concurrent migration'

def save():
    STATE['updated_at']=now()
    write_json(RUN/'reports/run_state.json',STATE)
    lines=['状态: '+STATE['status'],'阶段: '+STATE['phase'],'更新时间: '+STATE['updated_at'],
           '调度器 PID: '+str(os.getpid()),'评估最大并发: '+str(STATE.get('max_parallel_evaluations',3))]
    for name,_,_ in MODELS:
        e=STATE['evaluations'].get(name,{'status':'PENDING'})
        progress=''
        if e.get('stdout') and Path(e['stdout']).exists():
            n=len(re.findall(r'^Rollout \d+$',Path(e['stdout']).read_text(errors='replace'),re.M))
            progress=' ('+str(n)+'/30)'
        lines.append(name+': 训练='+STATE['models'][name]['status']+'; 评估='+e['status']+progress)
    if 'error' in STATE: lines.append('错误: '+STATE['error'])
    lines.append('finish.txt: '+str((RUN/'finish.txt').exists()))
    (RUN/'status.txt').write_text('\n'.join(lines)+'\n')

def monitor():
    quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip().split()
    sample={'time':now(),'phase':'accelerated_evaluation','loadavg':list(os.getloadavg()),
            'cpu_count':len(os.sched_getaffinity(0)),
            'cpu_quota_cores':float(quota[0])/float(quota[1]) if quota[0]!='max' else None,
            'children':{str(p.pid):p.poll() for p,_ in ACTIVE.values()}}
    try:
        output=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total',
                                        '--format=csv,noheader,nounits'],text=True,timeout=10)
        values=[float(v.strip()) for v in output.strip().split(',')]
        sample.update(gpu_utilization=values[0],memory_used_mib=values[1],memory_total_mib=values[2])
    except Exception as e: sample['monitor_error']=str(e)
    with (RUN/'logs/resource_monitor.jsonl').open('a') as f:f.write(json.dumps(sample)+'\n')
    return sample

def migrate():
    global STATE
    STATE=json.loads((RUN/'reports/run_state.json').read_text())
    assert STATE['phase']=='formal_evaluation' and STATE['status']=='RUNNING'
    assert all(STATE['models'][name]['status']=='SUCCEEDED' for name,_,_ in MODELS)
    assert json.loads((RUN/'reports/camera_filter_validation.json').read_text())['passed']
    assert not (RUN/'finish.txt').exists()
    stamp=str(time.time_ns())
    backup=RUN/'backups'/('evaluation_migration_'+stamp)
    backup.mkdir()
    shutil.copy2(RUN/'reports/run_state.json',backup/'run_state.before.json')
    stop_owned(STATE['pid'],'run_training_jobs.py')
    STATE=json.loads((RUN/'reports/run_state.json').read_text())
    history=STATE.setdefault('evaluation_history',[])
    for name,record in list(STATE['evaluations'].items()):
        if record['status']=='SUCCEEDED': continue
        stop_owned(record['pid'],'worker.py')
        old=Path(record['directory'])
        archive=old.with_name(old.name+'_before_acceleration_'+stamp)
        old.rename(archive)
        record.update(status='INTERRUPTED_FOR_ACCELERATION',end_time=now(),end_timestamp=time.time(),
                      exit_code=None,termination_signal='SIGTERM',previous_directory=str(old),directory=str(archive),
                      reason='User authorized parallel evaluation and removal of two unused camera renders')
        record['duration_seconds']=record['end_timestamp']-record['start_timestamp']
        record['stdout']=str(archive/'eval.stdout.log'); record['stderr']=str(archive/'eval.stderr.log')
        record['completed_rollouts']=len(re.findall(r'^Rollout \d+$',Path(record['stdout']).read_text(),re.M))
        write_json(archive/'job.acceleration_archive.json',record)
        history.append(dict(record,model=name))
        del STATE['evaluations'][name]
    old_lock=RUN/'pipeline.lock'
    if old_lock.exists(): old_lock.rename(backup/'pipeline.lock.previous')
    fd=os.open(str(old_lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,(str(os.getpid())+'\n').encode());os.close(fd)
    (RUN/'pipeline.pid').write_text(str(os.getpid())+'\n')
    STATE.update(pid=os.getpid(),phase='accelerated_formal_evaluation',max_parallel_evaluations=3,
                 acceleration_started_at=now(),acceleration_start_timestamp=time.time())
    STATE['acceleration']={'camera_filter_validation':str(RUN/'reports/camera_filter_validation.json'),
                           'max_parallel_evaluations':3,'torch_threads_per_worker':1,
                           'render_backend':'osmesa','top_camera_only':True,
                           'egl_status':'BLOCKED_MISSING_CONTAINER_GRAPHICS_CAPABILITY_AND_LIBRARIES',
                           'authorization':'User requested all three acceleration approaches together',
                           'legacy_evaluation_preserved':True}
    save()

def launch(name,policy,chunk):
    attempts=STATE.setdefault('accelerated_evaluation_attempts',{}).setdefault(name,[])
    directory=RUN/'evaluations'/(name if not attempts else name+'_accel_retry_'+str(len(attempts)))
    directory.mkdir(exist_ok=False)
    model=Path(STATE['models'][name]['directory'])
    for filename in ['policy_best.ckpt','dataset_stats.pkl']:(directory/filename).symlink_to(model/filename)
    cmd=[sys.executable,'-u',str(RUN/'scripts/worker.py'),'--phase','eval','--output',str(directory),
         '--policy',policy,'--chunk',str(chunk),'--epochs','1000','--num_eval_episodes','30',
         '--max_saved_videos','2','--top_camera_only','--torch_threads','1']
    legacy=next((r for r in STATE.get('evaluation_history',[]) if r.get('model')==name and r.get('completed_rollouts',0)>=2),None)
    if legacy:
        archive=Path(legacy['directory'])
        for i in range(2):(directory/('video'+str(i)+'.mp4')).symlink_to(archive/('video'+str(i)+'.mp4'))
        cmd+=['--reuse_videos_from',str(archive)]
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (directory/'eval.stdout.log').open('x') as out,(directory/'eval.stderr.log').open('x') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                              env=env,start_new_session=True)
    record={'label':name+'_accelerated_eval','status':'RUNNING','pid':proc.pid,'command':cmd,
            'directory':str(directory),'stdout':str(directory/'eval.stdout.log'),'stderr':str(directory/'eval.stderr.log'),
            'start_time':now(),'start_timestamp':time.time(),'render_backend':'osmesa','top_camera_only':True}
    attempts.append(record);STATE['evaluations'][name]=record
    (directory/'pid.txt').write_text(str(proc.pid)+'\n');write_json(directory/'job.json',record)
    ACTIVE[name]=(proc,record)
    print(now(),'START',name,proc.pid,flush=True);save()

def evaluate():
    queue=[(name,policy,chunk) for name,policy,chunk in MODELS if STATE['evaluations'].get(name,{}).get('status')!='SUCCEEDED']
    cap=3
    while queue or ACTIVE:
        while queue and len(ACTIVE)<cap:launch(*queue.pop(0))
        sample=monitor()
        for name,(proc,record) in list(ACTIVE.items()):
            if proc.poll() is None:continue
            record.update(exit_code=proc.returncode,end_time=now(),end_timestamp=time.time())
            record['duration_seconds']=record['end_timestamp']-record['start_timestamp']
            record['status']='SUCCEEDED' if proc.returncode==0 else 'FAILED'
            write_json(Path(record['directory'])/'job.json',record)
            del ACTIVE[name]
            if proc.returncode==0:
                metrics=parse_evaluation(record['directory'])
                assert json.loads((Path(record['directory'])/'worker_completion.json').read_text())['completed']
                record['metrics']=metrics
                write_json(Path(record['directory'])/'evaluation_metrics.json',metrics)
                print(now(),'SUCCESS',name,metrics['successes'],'/30',flush=True)
            else:
                error=Path(record['stderr']).read_text(errors='replace')
                if 'out of memory' in error.lower() and len(STATE['accelerated_evaluation_attempts'][name])<3:
                    cap=max(1,cap-1);STATE['max_parallel_evaluations']=cap
                    record['retry_reason']='OOM; reduce concurrency; restart in fresh evaluation directory'
                    queue.insert(0,next(m for m in MODELS if m[0]==name))
                else:raise RuntimeError(name+' accelerated evaluation failed: '+record['stderr'])
        save();time.sleep(5)

def main():
    try:
        migrate();evaluate()
        STATE['phase']='collect_plot_report_validate';save()
        for script in ['collect_results.py','plot_results.py','finalize.py']:
            subprocess.run([sys.executable,str(RUN/'scripts'/script)],cwd=ROOT,check=True)
        STATE.update(status='COMPLETED',phase='completed',end_time=now());save()
        print(now(),'ALL_COMPLETED',flush=True)
    except BaseException as exc:
        for name,(proc,record) in ACTIVE.items():
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=25)
                record.update(status='INTERRUPTED',exit_code=proc.returncode,end_time=now())
                record['duration_seconds']=time.time()-record['start_timestamp']
                write_json(Path(record['directory'])/'job.json',record)
        if STATE is not None:
            STATE.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc());save()
            failure=RUN/'reports'/('acceleration_failure_'+str(time.time_ns())+'.md')
            failure.write_text('# 加速流程未完成\n\n'+traceback.format_exc()+'\n未创建成功标志。\n')
        raise

if __name__=='__main__':main()
