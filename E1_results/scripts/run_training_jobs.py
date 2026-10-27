"""Persistent E1/E3 pipeline. Never resumes optimizer state or reuses failed output directories."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from common import ROOT, RUN, MODELS, now, write_json, validate_artifacts, parse_evaluation

STATE = {'started_at':now(),'pid':os.getpid(),'python':sys.executable,'phase':'initializing',
         'status':'RUNNING','models':{},'tests':{},'evaluations':{},'max_parallel':3}
CHILDREN = {}

def save():
    STATE['updated_at'] = now()
    write_json(RUN/'reports/run_state.json',STATE)
    text = '状态: '+STATE['status']+'\n阶段: '+STATE['phase']+'\n更新时间: '+STATE['updated_at']+'\n'
    text += '调度器 PID: '+str(os.getpid())+'\n'
    for name,_,_ in MODELS:
        text += name+': 训练='+STATE['models'].get(name,{}).get('status','PENDING')
        text += '; 评估='+STATE['evaluations'].get(name,{}).get('status','PENDING')+'\n'
    if 'error' in STATE: text += '错误: '+STATE['error']+'\n'
    text += '剩余任务: '+ ('无' if STATE['status']=='COMPLETED' else '按 run_state.json 完成当前阶段及后续训练、评估、汇总、完整性检查')+'\n'
    (RUN/'status.txt').write_text(text)
    print(now(),STATE['phase'],STATE['status'],flush=True)

def monitor():
    sample = {'time':now(),'loadavg':list(os.getloadavg()),'cpu_count':len(os.sched_getaffinity(0)),
              'children':{str(p.pid):p.poll() for p in CHILDREN.values()}}
    try:
        output = subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total',
                                          '--format=csv,noheader,nounits'],text=True,timeout=15)
        values = [float(v.strip()) for v in output.strip().split(',')]
        sample.update(gpu_utilization=values[0],memory_used_mib=values[1],memory_total_mib=values[2])
        sample['gpu_processes'] = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
                                                          '--format=csv,noheader,nounits'],text=True,timeout=15).strip()
    except Exception as e: sample['monitor_error'] = str(e)
    with (RUN/'logs/resource_monitor.jsonl').open('a') as f: f.write(json.dumps(sample)+'\n')
    return sample

def start_job(label, policy, chunk, epochs, directory, phase='train'):
    directory.mkdir(parents=True,exist_ok=False)
    if phase=='eval':
        model_name = label.removesuffix('_eval') if hasattr(label,'removesuffix') else label[:-5]
        model = Path(STATE['models'][model_name]['directory'])
        for file in ['policy_best.ckpt','dataset_stats.pkl']:
            (directory/file).symlink_to(model/file)
    cmd = [sys.executable,'-u',str(RUN/'scripts/worker.py'),'--phase',phase,'--output',str(directory),
           '--policy',policy,'--chunk',str(chunk),'--epochs',str(epochs)]
    if phase=='eval': cmd += ['--num_eval_episodes','30','--max_saved_videos','2']
    out = directory/('train.stdout.log' if phase=='train' else 'eval.stdout.log')
    err = directory/('train.stderr.log' if phase=='train' else 'eval.stderr.log')
    with out.open('x') as stdout,err.open('x') as stderr:
        proc = subprocess.Popen(cmd,cwd=ROOT,stdout=stdout,stderr=stderr,start_new_session=True)
    rec = {'label':label,'status':'RUNNING','pid':proc.pid,'command':cmd,'directory':str(directory),
           'stdout':str(out),'stderr':str(err),'start_time':now(),'start_timestamp':time.time()}
    CHILDREN[label] = proc
    (directory/'pid.txt').write_text(str(proc.pid)+'\n')
    write_json(directory/'job.json',rec)
    return proc,rec

def end_job(proc, rec):
    rec.update(exit_code=proc.returncode,end_time=now(),end_timestamp=time.time())
    rec['duration_seconds'] = rec['end_timestamp']-rec['start_timestamp']
    rec['status'] = 'SUCCEEDED' if proc.returncode==0 else 'FAILED'
    write_json(Path(rec['directory'])/'job.json',rec)

def wait_job(proc, rec):
    while proc.poll() is None:
        monitor(); time.sleep(5)
    end_job(proc,rec)
    if proc.returncode: raise RuntimeError(rec['label']+' failed: '+rec['stderr'])

def stage_test(name,policy,chunk,epochs):
    STATE['phase'] = name; save()
    existing = RUN/'tests'/name
    if (existing/'artifact_validation.json').exists():
        artifacts = validate_artifacts(existing,epochs)
        assert epochs==2, 'Only completed smoke checks may be recovered automatically'
        rec = {'label':name,'status':'SUCCEEDED','directory':str(existing),'artifacts':artifacts,
               'recovered':True,'reason':'Detached worker completed; initial launcher exited before persisting job metadata',
               'timing_source':'filesystem mtimes; smoke check only',
               'start_timestamp':(existing/'training_config.json').stat().st_mtime,
               'end_timestamp':(existing/'artifact_validation.json').stat().st_mtime,
               'stdout':str(existing/'train.stdout.log'),'stderr':str(existing/'train.stderr.log')}
        rec['duration_seconds']=rec['end_timestamp']-rec['start_timestamp']
        STATE['tests'][name]=rec; save()
        return rec
    proc,rec = start_job(name,policy,chunk,epochs,RUN/'tests'/name)
    STATE['tests'][name] = rec; save()
    wait_job(proc,rec)
    rec['artifacts'] = validate_artifacts(rec['directory'],epochs)
    save()
    return rec

def train_formal(benchmark):
    STATE['phase'] = 'formal_training'; save()
    queue = list(MODELS)
    active = {}
    cap = 3
    contention_count = 0
    while queue or active:
        while queue and len(active)<cap:
            name,policy,chunk = queue.pop(0)
            record = STATE['models'].setdefault(name,{'status':'PENDING','attempts':[]})
            attempt = len(record['attempts'])
            directory = RUN/'models'/name if attempt==0 else RUN/'models'/(name+'_retry_'+str(attempt))
            proc,rec = start_job(name,policy,chunk,1000,directory)
            record['attempts'].append(rec); record.update(status='RUNNING',directory=str(directory))
            active[name] = (proc,rec,policy,chunk); save()
        sample = monitor()
        for name,(proc,rec,policy,chunk) in list(active.items()):
            if proc.poll() is None: continue
            end_job(proc,rec)
            del active[name]
            record = STATE['models'][name]
            if proc.returncode==0:
                record['artifacts'] = validate_artifacts(rec['directory'],1000)
                record['status'] = 'SUCCEEDED'
            else:
                error = Path(rec['stderr']).read_text(errors='replace')
                if ('out of memory' in error.lower() or rec.get('retry_reason')) and len(record['attempts'])<3:
                    cap = max(1,cap-1)
                    rec['retry_reason'] = rec.get('retry_reason','CUDA OOM; reduce concurrency and restart in fresh directory')
                    record['status'] = 'RETRY_PENDING'
                    queue.insert(0,(name,policy,chunk))
                else:
                    record['status']='FAILED'; save()
                    raise RuntimeError(name+' failed; see '+rec['stderr'])
            STATE['max_parallel'] = cap; save()
        # Sustained near-exhaustion, or slow epochs with a heavily oversubscribed CPU,
        # reduces concurrency by stopping only a subprocess created by this scheduler.
        high_mem = sample.get('memory_used_mib',0)>sample.get('memory_total_mib',81920)*0.96
        cpu_busy = sample['loadavg'][0]>sample['cpu_count']*1.5
        severe_slow = False
        for name,(proc,rec,policy,chunk) in active.items():
            if policy!='ACT': continue
            metrics_path = Path(rec['directory'])/'training_metrics.json'
            if metrics_path.exists() and time.time()-rec['start_timestamp']>300:
                n = json.loads(metrics_path.read_text())['epochs_completed']
                if n>100 and (time.time()-rec['start_timestamp'])/n>benchmark*6: severe_slow=True
        contention_count = contention_count+1 if (high_mem or (cpu_busy and severe_slow)) else 0
        if contention_count>=6 and len(active)>1 and cap>1:
            name = next(reversed(active))
            proc,rec,_,_ = active[name]
            rec['retry_reason'] = 'Sustained resource contention; reduce concurrency and restart in independent directory'
            os.killpg(proc.pid,signal.SIGTERM)
            cap -= 1
            STATE['max_parallel'] = cap; contention_count=0; save()
        time.sleep(5)

def evaluate():
    STATE['phase']='formal_evaluation'; save()
    for name,policy,chunk in MODELS:
        directory = RUN/'evaluations'/name
        proc,rec = start_job(name+'_eval',policy,chunk,1000,directory,'eval')
        STATE['evaluations'][name] = rec; save()
        wait_job(proc,rec)
        rec['metrics'] = parse_evaluation(directory)
        write_json(directory/'evaluation_metrics.json',rec['metrics'])
        save()

def main():
    lock = RUN/'pipeline.lock'
    fd = os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,(str(os.getpid())+'\n').encode()); os.close(fd)
    (RUN/'pipeline.pid').write_text(str(os.getpid())+'\n')
    try:
        assert json.loads((RUN/'reports/preflight.json').read_text())['passed']
        save()
        stage_test('smoke_ACT_k100','ACT',100,2)
        stage_test('smoke_CNNMLP','CNNMLP',1,2)
        b = stage_test('benchmark_ACT_k100_50epochs','ACT',100,50)
        per_epoch = b['duration_seconds']/50
        # Conservative serial estimate: ACT benchmark applied to all four models.
        estimate = {'benchmark_seconds':b['duration_seconds'],'seconds_per_epoch_including_startup':per_epoch,
                    'estimated_serial_training_seconds':per_epoch*4000,
                    'estimated_parallel_lower_bound_seconds':per_epoch*4000/3,
                    'over_24_hours':per_epoch*4000>86400,
                    'note':'实测 ACT 50 轮外推；CNNMLP、并行争用、I/O 和评估耗时另计。资源数据见 resource_monitor.jsonl。'}
        write_json(RUN/'reports/timing_estimate.json',estimate)
        (RUN/'reports/timing_estimate.md').write_text('50 轮测速耗时 %.2f 秒，平均 %.3f 秒/轮。四模型保守串行训练估计 %.2f 小时；三进程理想下界 %.2f 小时。%s\n'%(
            b['duration_seconds'],per_epoch,per_epoch*4000/3600,per_epoch*4000/3/3600,
            '估计超过 24 小时，按授权自动继续。' if estimate['over_24_hours'] else '按授权自动继续。'))
        train_start=time.time()
        train_formal(per_epoch)
        STATE['formal_training_wall_seconds'] = time.time()-train_start
        STATE['formal_training_sum_process_seconds'] = sum(a['duration_seconds'] for m in STATE['models'].values() for a in m['attempts'])
        save()
        evaluate()
        STATE['phase']='collect_plot_report_validate'; save()
        for script in ['collect_results.py','plot_results.py','finalize.py']:
            subprocess.run([sys.executable,str(RUN/'scripts'/script)],cwd=ROOT,check=True)
        STATE['status']='COMPLETED'; STATE['phase']='completed'; STATE['end_time']=now(); save()
    except BaseException as exc:
        STATE['status']='FAILED'; STATE['error']=repr(exc)
        STATE['traceback']=traceback.format_exc()
        # Only stop this scheduler's still-running children; never touch unrelated server processes.
        for label,proc in CHILDREN.items():
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                proc.wait(timeout=30)
                for model in STATE['models'].values():
                    for attempt in model['attempts']:
                        if attempt['pid']==proc.pid:
                            end_job(proc,attempt); attempt['stop_reason']='Pipeline failure'; model['status']='INTERRUPTED'
        save()
        (RUN/'reports/failure_report.md').write_text('# 实验未完成\n\n'+STATE['error']+'\n\n```\n'+STATE['traceback']+'\n```\n\n未创建 finish.txt。详见 status.txt、run_state.json 和独立任务日志。\n')
        raise

if __name__=='__main__':
    main()
