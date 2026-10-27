"""Persistent, bounded V2 supervisor; all partial attempts remain reviewable."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from common import (ROOT,RUN,PYTHON,MODELS,EVALS,TRAINING_SEEDS,now,write_json,worker_environment,
                    validate_model,check_sources,sha256)

STATE={}
CHILDREN={}
LAST_MONITOR=0.


class AdoptedWorker:
    """Observe a live worker without restarting it or claiming its OS exit code."""
    def __init__(self,rec):
        self.pid=rec['pid'];self.directory=Path(rec['directory']);self.returncode=None
        self.start_ticks=None
        path=Path('/proc/%d'%self.pid)
        if path.exists():
            fields=(path/'stat').read_text().rsplit(')',1)[1].split()
            if fields[0]!='Z':
                assert path.stat().st_uid==os.getuid()
                command=(path/'cmdline').read_bytes().split(b'\0')
                assert str(RUN/'scripts/worker.py').encode() in command
                assert str(self.directory).encode() in command
                self.start_ticks=fields[19]
        rec.update(adopted_without_restart=True,adopted_at=now(),worker_start_ticks=self.start_ticks)

    def poll(self):
        if self.returncode is not None:return self.returncode
        try:
            fields=Path('/proc/%d/stat'%self.pid).read_text().rsplit(')',1)[1].split()
            if self.start_ticks is not None and fields[19]==self.start_ticks and fields[0]!='Z':return None
        except FileNotFoundError:pass
        marker=self.directory/'worker_completion.json'
        self.returncode=0 if marker.exists() and json.loads(marker.read_text()).get('completed') else 1
        return self.returncode

    def wait(self,timeout=None):
        deadline=time.monotonic()+timeout if timeout else float('inf')
        while self.poll() is None:
            if time.monotonic()>deadline:raise subprocess.TimeoutExpired('adopted worker',timeout)
            time.sleep(.1)
        return self.returncode


def save():
    STATE['updated_at']=now()
    write_json(RUN/'reports/run_state.json',STATE)
    lines=['V2 '+STATE['status'],'阶段: '+STATE['phase'],'更新时间: '+STATE['updated_at'],
        '正式训练种子: '+str(TRAINING_SEEDS)+'；模型 '+str(len(MODELS))+' 个；评估 '+str(len(EVALS))+' 组 / '+str(50*len(EVALS))+' 回合',
        '渲染: '+STATE.get('renderer','llvmpipe')+' / OSMesa CPU；NVIDIA EGL 未启用（用户已授权 CPU 回退）',
        'ACT 训练使用 A800 CUDA GPU；渲染使用 CPU',
        '调度器 PID: '+str(os.getpid()),'训练并发: '+str(STATE.get('train_parallel','测速后决定')),
        '评估并发: '+str(STATE.get('eval_parallel','测速后决定')),
        '训练期间评估并发: '+str(STATE.get('evaluations_during_training',0))]
    for section,specs,progress_key in [('models',MODELS,'epochs_completed'),('evaluations',EVALS,'episodes_completed')]:
        lines.append('训练任务' if section=='models' else '评估任务')
        for spec in specs:
            rec=STATE[section][spec['name']]
            progress=''
            if rec.get('directory'):
                path=Path(rec['directory'])/'progress.json'
                if path.exists():
                    data=json.loads(path.read_text())
                    rec['progress']=data
                    progress=' %s/%s'%(data.get(progress_key,0),2000 if section=='models' else 50)
            lines.append('  '+spec['name']+': '+rec['status']+progress)
    if 'error' in STATE: lines.append('错误: '+STATE['error'])
    lines.append('finish.txt: '+str((RUN/'finish.txt').exists()))
    temp=RUN/'status.txt.tmp';temp.write_text('\n'.join(lines)+'\n');os.replace(str(temp),str(RUN/'status.txt'))
    write_json(RUN/'reports/run_state.json',STATE)


def monitor(force=False):
    global LAST_MONITOR
    if not force and time.monotonic()-LAST_MONITOR<15:
        return
    LAST_MONITOR=time.monotonic()
    row=dict(time=now(),phase=STATE['phase'],loadavg=list(os.getloadavg()),
        cpu_stat=Path('/sys/fs/cgroup/cpu.stat').read_text(),
        cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        memory_current_bytes=int(Path('/sys/fs/cgroup/memory.current').read_text()),workers={})
    for name,p in CHILDREN.items():
        if p.poll() is None:
            try:
                fields=Path('/proc/%d/stat'%p.pid).read_text().rsplit(')',1)[1].split()
                row['workers'][name]=dict(pid=p.pid,utime_ticks=int(fields[11]),stime_ticks=int(fields[12]),
                    rss_pages=int(fields[21]),clock_ticks_per_second=os.sysconf('SC_CLK_TCK'))
            except FileNotFoundError: pass
    try:
        row['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'],text=True,timeout=10).strip()
        row['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
            '--format=csv,noheader'],text=True,timeout=10).strip()
    except Exception as exc: row['gpu_query_error']=repr(exc)
    with (RUN/'logs/resource_monitor.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')


def fresh_directory(base):
    if not base.exists():
        base.mkdir(parents=True)
        return base
    suffix=1
    while base.with_name(base.name+'_attempt%d'%suffix).exists(): suffix+=1
    result=base.with_name(base.name+'_attempt%d'%suffix);result.mkdir(parents=True)
    return result


def spawn(label,spec,directory,phase='train',epochs=2000,gate=None,smoke=None,model=None):
    if phase=='eval':
        for filename in ['policy_best.ckpt','dataset_stats.pkl']:
            (directory/filename).symlink_to(model/filename)
    command=[PYTHON,'-u',str(RUN/'scripts/worker.py'),'--phase',phase,'--output',str(directory),
        '--chunk',str(spec['chunk']),'--seed',str(spec['seed']),'--epochs',str(epochs)]
    if spec.get('temporal'):command.append('--temporal')
    if gate:command+=['--gate',str(gate)]
    if smoke:command+=['--smoke_episodes',str(smoke)]
    with (directory/'stdout.log').open('x') as out,(directory/'stderr.log').open('x') as err:
        proc=subprocess.Popen(command,cwd=str(ROOT),env=worker_environment(),stdout=out,stderr=err,start_new_session=True)
    rec=dict(label=label,status='RUNNING',pid=proc.pid,command=command,directory=str(directory),
        started_at=now(),start_timestamp=time.time(),start_monotonic=time.monotonic())
    CHILDREN[label]=proc
    write_json(directory/'job.json',rec)
    (directory/'pid.txt').write_text(str(proc.pid)+'\n')
    return proc,rec


def end(proc,rec):
    rec.update(exit_code=proc.returncode,ended_at=now(),end_timestamp=time.time(),
        elapsed_seconds=time.monotonic()-rec['start_monotonic'],status='SUCCEEDED' if proc.returncode==0 else 'FAILED')
    if isinstance(proc,AdoptedWorker):
        rec.update(exit_code=None,completion_status_source='worker completion artifact and subsequent validation; OS exit code unavailable after supervisor handoff',
                   artifact_completion_detected=proc.returncode==0)
    write_json(Path(rec['directory'])/'job.json',rec)
    CHILDREN.pop(rec['label'],None)


def wait_all(jobs,timeout=None):
    start=time.monotonic()
    while any(p.poll() is None for p,_ in jobs):
        monitor();save()
        if timeout and time.monotonic()-start>timeout:
            raise RuntimeError('Own job timeout; logs preserved')
        time.sleep(5)
    for proc,rec in jobs:end(proc,rec)


def train_benchmarks():
    STATE['phase']='training_throughput_1_2_3';save()
    results=[]
    for concurrency in [1,2,3]:
        base=fresh_directory(RUN/'tests'/('training_throughput_%d'%concurrency))
        gate=base/'start.gate';jobs=[]
        for seed in range(concurrency):
            directory=base/('seed%d'%seed);directory.mkdir()
            jobs.append(spawn('benchmark_c%d_s%d'%(concurrency,seed),dict(chunk=100,seed=seed),directory,epochs=30,gate=gate))
        deadline=time.monotonic()+300
        while not all((Path(rec['directory'])/'ready.txt').exists() for _,rec in jobs):
            assert all(p.poll() is None for p,_ in jobs),'Benchmark failed before gate; inspect logs'
            assert time.monotonic()<deadline,'Benchmark worker initialization timed out'
            monitor();save();time.sleep(2)
        start=time.monotonic();gate.write_text(now())
        wait_all(jobs,timeout=900)
        wall=time.monotonic()-start
        rec=dict(concurrency=concurrency,epochs_per_process=30,wall_seconds=wall,
            all_processes_succeeded=all(p.returncode==0 for p,_ in jobs),directories=[r['directory'] for _,r in jobs])
        if rec['all_processes_succeeded']:
            series=[[json.loads(line) for line in (Path(r['directory'])/'epoch_metrics.jsonl').read_text().splitlines()] for _,r in jobs]
            assert all(len(rows)==30 for rows in series)
            begin=max(rows[5]['epoch_start_monotonic'] for rows in series)
            finish=min(rows[-1]['epoch_end_monotonic'] for rows in series)
            count=sum(sum(row['epoch_start_monotonic']>=begin and row['epoch_end_monotonic']<=finish for row in rows) for rows in series)
            rec.update(total_epochs_per_wall_second=concurrency*30/wall,
                steady_overlap_seconds=finish-begin,steady_completed_epochs=count,
                steady_total_epochs_per_wall_second=count/(finish-begin),
                mean_epoch_seconds_by_worker=[sum(x['epoch_seconds'] for x in rows[5:])/25 for rows in series])
        else:
            errors='\n'.join((Path(r['directory'])/'stderr.log').read_text(errors='replace') for _,r in jobs)
            assert 'out of memory' in errors.lower(),'Non-OOM benchmark failure; do not proceed'
        results.append(rec)
        write_json(RUN/'reports/training_throughput.json',dict(results=results,status='measuring'))
    good=[r for r in results if r['all_processes_succeeded']]
    assert good
    # Real overlapping completed epochs per wall second, after five warmup epochs.
    best=max(good,key=lambda r:r['steady_total_epochs_per_wall_second'])
    STATE['train_parallel']=best['concurrency']
    write_json(RUN/'reports/training_throughput.json',dict(results=results,status='complete',
        selected_concurrency=STATE['train_parallel'],selection='Highest measured steady overlapping completed epochs per wall second; full wall throughput also recorded',
        estimated_training_hours=len(MODELS)*2000/best['steady_total_epochs_per_wall_second']/3600,
        note='Estimate excludes checkpoint I/O, validation at completion, retries, and evaluation'))
    save()


def evaluation_benchmark():
    STATE['phase']='evaluation_smoke_and_throughput';save()
    model=RUN/'tests/smoke_training_k100_seed0'
    assert json.loads((model/'worker_completion.json').read_text())['completed']
    results=[]
    for count in [1,2]:
        jobs=[];start=time.monotonic()
        for seed in range(count):
            directory=fresh_directory(RUN/'tests'/('eval_throughput_%d_worker%d'%(count,seed)))
            jobs.append(spawn('eval_bench_%d_%d'%(count,seed),dict(chunk=100,seed=0,temporal=True),
                directory,phase='eval',epochs=3,smoke=1,model=model))
        wait_all(jobs,timeout=600)
        assert all(p.returncode==0 for p,_ in jobs),'Evaluation smoke failed'
        wall=time.monotonic()-start
        results.append(dict(concurrency=count,episodes=count,wall_seconds=wall,total_episodes_per_second=count/wall,
            directories=[r['directory'] for _,r in jobs]))
    STATE['eval_parallel']=max(results,key=lambda x:x['total_episodes_per_second'])['concurrency']
    # Also exercise unaggregated official evaluation before formal work.
    directory=fresh_directory(RUN/'tests/eval_smoke_no_temporal')
    jobs=[spawn('eval_smoke_no_temporal',dict(chunk=100,seed=0,temporal=False),directory,
                phase='eval',epochs=3,smoke=1,model=model)]
    wait_all(jobs,timeout=600)
    assert jobs[0][0].returncode==0
    write_json(RUN/'reports/evaluation_throughput.json',dict(results=results,selected_concurrency=STATE['eval_parallel'],
        temporal_and_non_temporal_smoke_passed=True,note='Preliminary 3-epoch policy, one episode each; excluded from all formal statistics'))
    save()


def train_formal():
    STATE['phase']='formal_training';save()
    queue=[m for m in MODELS if STATE['models'][m['name']]['status']!='SUCCEEDED']
    active={};cap=STATE['train_parallel']
    while queue or active:
        while queue and len(active)<cap:
            spec=queue.pop(0);name=spec['name'];record=STATE['models'][name]
            directory=fresh_directory(RUN/'models'/name)
            proc,rec=spawn(name,spec,directory)
            record.setdefault('attempts',[]).append(rec);record.update(status='RUNNING',directory=str(directory),pid=proc.pid)
            active[name]=(proc,rec,spec);save()
        monitor()
        for name,(proc,rec,spec) in list(active.items()):
            if proc.poll() is None:continue
            end(proc,rec);del active[name]
            record=STATE['models'][name]
            if proc.returncode==0:
                record['artifacts']=json.loads((Path(rec['directory'])/'artifact_validation.json').read_text())
                assert record['artifacts']['epochs']==2000
                record['status']='SUCCEEDED'
            else:
                error=(Path(rec['directory'])/'stderr.log').read_text(errors='replace')
                if 'out of memory' in error.lower() and cap>1 and len(record['attempts'])<3:
                    cap-=1;STATE['train_parallel']=cap
                    rec['retry_reason']='CUDA OOM; fresh random initialization in new directory, lower concurrency'
                    record['status']='RETRY_PENDING';queue.insert(0,spec)
                else:
                    record['status']='FAILED';save();raise RuntimeError('Training failed: '+name)
            save()
        save();time.sleep(5)


def evaluate_formal():
    STATE['phase']='formal_evaluation';save()
    queue=[e for e in EVALS if STATE['evaluations'][e['name']]['status']!='SUCCEEDED'];active={}
    while queue or active:
        while queue and len(active)<STATE['eval_parallel']:
            spec=queue.pop(0);name=spec['name'];record=STATE['evaluations'][name]
            model=Path(STATE['models'][spec['model']]['directory'])
            directory=fresh_directory(RUN/'evaluations'/name)
            proc,rec=spawn(name,spec,directory,phase='eval',model=model)
            record.setdefault('attempts',[]).append(rec);record.update(status='RUNNING',directory=str(directory),pid=proc.pid)
            active[name]=(proc,rec);save()
        monitor()
        for name,(proc,rec) in list(active.items()):
            if proc.poll() is None:continue
            end(proc,rec);del active[name]
            record=STATE['evaluations'][name]
            if proc.returncode:
                record['status']='FAILED';save();raise RuntimeError('Evaluation failed: '+name)
            metrics=json.loads((Path(rec['directory'])/'evaluation_metrics.json').read_text())
            assert metrics['episodes']==50 and metrics['context_released']
            record.update(status='SUCCEEDED',metrics=metrics);save()
            subprocess.run([PYTHON,str(RUN/'scripts/collect_results.py'),'--partial'],cwd=str(ROOT),
                           env=worker_environment(),check=True)
        save();time.sleep(5)


def parallel_limits():
    path=RUN/'parallel_controls.json'
    data=json.loads(path.read_text()) if path.exists() else {}
    train=int(data.get('train_parallel',STATE['train_parallel']))
    evaluation=int(data.get('eval_parallel',STATE['eval_parallel']))
    overlap=int(data.get('evaluations_during_training',1))
    assert 1<=train<=3 and 1<=evaluation<=3 and 0<=overlap<=evaluation
    STATE.update(train_parallel=train,eval_parallel=evaluation,evaluations_during_training=overlap)
    return train,evaluation,overlap


def run_overlapped():
    """Train pending models and evaluate completed ones, preserving all live jobs."""
    STATE['phase']='overlapped_training_and_evaluation'
    active_train={};active_eval={}
    for section,active,specs in [('models',active_train,MODELS),('evaluations',active_eval,EVALS)]:
        for spec in specs:
            record=STATE[section][spec['name']]
            if record['status']=='RUNNING':
                rec=record['attempts'][-1]
                proc=AdoptedWorker(rec)
                active[spec['name']]=(proc,rec,spec);CHILDREN[spec['name']]=proc
    queue_train=[m for m in MODELS if STATE['models'][m['name']]['status'] not in ['RUNNING','SUCCEEDED']]
    queue_eval=[e for e in EVALS if STATE['evaluations'][e['name']]['status'] not in ['RUNNING','SUCCEEDED']]
    save()
    while queue_train or queue_eval or active_train or active_eval:
        train_cap,eval_cap,overlap_cap=parallel_limits()
        while queue_train and len(active_train)<train_cap:
            spec=queue_train.pop(0);name=spec['name'];record=STATE['models'][name]
            directory=fresh_directory(RUN/'models'/name)
            proc,rec=spawn(name,spec,directory)
            record.setdefault('attempts',[]).append(rec)
            record.update(status='RUNNING',directory=str(directory),pid=proc.pid)
            active_train[name]=(proc,rec,spec);save()
        current_eval_cap=overlap_cap if queue_train or active_train else eval_cap
        while len(active_eval)<current_eval_cap:
            spec=next((e for e in queue_eval if STATE['models'][e['model']]['status']=='SUCCEEDED'),None)
            if spec is None:break
            queue_eval.remove(spec);name=spec['name'];record=STATE['evaluations'][name]
            directory=fresh_directory(RUN/'evaluations'/name)
            model=Path(STATE['models'][spec['model']]['directory'])
            proc,rec=spawn(name,spec,directory,phase='eval',model=model)
            record.setdefault('attempts',[]).append(rec)
            record.update(status='RUNNING',directory=str(directory),pid=proc.pid)
            active_eval[name]=(proc,rec,spec);save()
        monitor()
        for name,(proc,rec,spec) in list(active_train.items()):
            if proc.poll() is None:continue
            end(proc,rec);del active_train[name];record=STATE['models'][name]
            if proc.returncode==0:
                artifacts=json.loads((Path(rec['directory'])/'artifact_validation.json').read_text())
                assert artifacts['epochs']==2000
                record.update(status='SUCCEEDED',artifacts=artifacts)
            else:
                error=(Path(rec['directory'])/'stderr.log').read_text(errors='replace')
                if 'out of memory' in error.lower() and train_cap>1 and len(record['attempts'])<3:
                    train_cap-=1
                    controls=json.loads((RUN/'parallel_controls.json').read_text())
                    controls.update(train_parallel=train_cap,reason='CUDA OOM; reduce concurrency, preserve failed attempt and restart from scratch')
                    write_json(RUN/'parallel_controls.json',controls)
                    record['status']='RETRY_PENDING';queue_train.insert(0,spec)
                else:
                    record['status']='FAILED';save();raise RuntimeError('Training failed: '+name)
            save()
        for name,(proc,rec,spec) in list(active_eval.items()):
            if proc.poll() is None:continue
            end(proc,rec);del active_eval[name];record=STATE['evaluations'][name]
            if proc.returncode:
                record['status']='FAILED';save();raise RuntimeError('Evaluation failed: '+name)
            metrics=json.loads((Path(rec['directory'])/'evaluation_metrics.json').read_text())
            assert metrics['episodes']==50 and metrics['context_released']
            record.update(status='SUCCEEDED',metrics=metrics);save()
            subprocess.run([PYTHON,str(RUN/'scripts/collect_results.py'),'--partial'],cwd=str(ROOT),
                           env=worker_environment(),check=True)
        save();time.sleep(5)


def main():
    global STATE
    parser=argparse.ArgumentParser();parser.add_argument('--continue_run',action='store_true')
    parser.add_argument('--adopt_running',action='store_true');args=parser.parse_args()
    assert not args.adopt_running or args.continue_run
    lock=(RUN/'pipeline.lock').open('a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
    if args.continue_run:
        STATE=json.loads((RUN/'reports/run_state.json').read_text())
        assert STATE['status']!='COMPLETED'
        for section in ['models','evaluations']:
            for record in STATE[section].values():
                if record['status']=='RUNNING':
                    if args.adopt_running:
                        # Ownership/command/start-time validation occurs in AdoptedWorker.
                        continue
                    pid=record.get('pid')
                    if pid and Path('/proc/%d'%pid).exists():
                        command=(Path('/proc/%d/cmdline'%pid)).read_bytes().replace(b'\0',b' ').decode()
                        assert str(RUN/'scripts/worker.py') not in command,'Prior worker still alive; do not duplicate'
                    record['status']='INTERRUPTED'
        STATE.pop('error',None);STATE.pop('traceback',None)
    else:
        assert not (RUN/'reports/run_state.json').exists(),'Use --continue_run; never overwrite run state'
        STATE=dict(status='RUNNING',phase='validating',started_at=now(),models={m['name']:dict(status='PENDING') for m in MODELS},
            evaluations={e['name']:dict(status='PENDING') for e in EVALS},renderer='llvmpipe')
    STATE.update(pid=os.getpid(),status='RUNNING')
    (RUN/'pipeline.pid').write_text(str(os.getpid())+'\n')
    try:
        assert not (RUN/'finish.txt').exists()
        check_sources()
        assert json.loads((RUN/'reports/preflight.json').read_text())['passed']
        selection=json.loads((RUN/'reports/render_selection.json').read_text())
        assert selection['camera_filter_validated']
        assert json.loads((RUN/'tests/render_llvmpipe/validation.json').read_text())['passed']
        assert json.loads((RUN/'reports/software_renderer_comparison.json').read_text())['physical_states_exact']
        save();monitor(force=True)
        if 'train_parallel' not in STATE:train_benchmarks()
        if 'eval_parallel' not in STATE:evaluation_benchmark()
        # Prerequisite smoke runs and throughput jobs are never formal models.
        if args.adopt_running or (RUN/'parallel_controls.json').exists():
            run_overlapped()
        else:
            train_formal();evaluate_formal()
        STATE['phase']='final_integrity_and_report';save()
        subprocess.run([PYTHON,str(RUN/'scripts/collect_results.py')],cwd=str(ROOT),env=worker_environment(),check=True)
        STATE.update(status='COMPLETED',phase='completed',ended_at=now());save()
    except BaseException as exc:
        STATE.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
        for label,proc in list(CHILDREN.items()):
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:pass
                for section in ['models','evaluations']:
                    if label in STATE[section]:STATE[section][label]['status']='INTERRUPTED'
        save()
        (RUN/'reports/failure_report.md').write_text('V2 尚未完成，未创建 finish.txt。\n\n'+STATE['traceback']+
            '\n失败/中断目录已保留；不能从仅有权重的检查点恢复优化器。使用 --continue_run 将在新目录从头重跑未完成模型。\n')
        raise


if __name__=='__main__':
    main()
