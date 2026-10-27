"""Measure a third worker alongside live training, without stopping formal jobs."""
import json
import subprocess
import time
from pathlib import Path
from common import ROOT,RUN,PYTHON,now,write_json,worker_environment


def rows(path):
    data=path.read_text()
    return [json.loads(line) for line in data.splitlines() if line and line.endswith('}')]


def completed(directory,start,end):
    return sum(r['epoch_start_monotonic']>=start and r['epoch_end_monotonic']<=end
               for r in rows(directory/'epoch_metrics.jsonl'))


def sample(path,phase):
    row=dict(time=now(),phase=phase,cpu_stat=Path('/sys/fs/cgroup/cpu.stat').read_text())
    try:
        row['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'],text=True,timeout=10).strip()
    except Exception as e:row['error']=repr(e)
    with path.open('a') as f:f.write(json.dumps(row)+'\n')


def main():
    state=json.loads((RUN/'reports/run_state.json').read_text())
    active={name:Path(rec['directory']) for name,rec in state['models'].items() if rec['status']=='RUNNING'}
    assert len(active)==2 and state['phase']=='formal_training'
    output=RUN/'tests'/('live_parallel_'+time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir()
    write_json(RUN/'reports/live_parallel_tuning.json',dict(status='measuring_two_workers',directory=str(output),started_at=now()))
    start=time.perf_counter()
    for _ in range(9):sample(output/'resources.jsonl','two_workers');time.sleep(10)
    end=time.perf_counter()
    baseline=dict(window_seconds=end-start,workers={k:completed(p,start,end) for k,p in active.items()})
    baseline['total_epochs_per_second']=sum(baseline['workers'].values())/(end-start)
    write_json(output/'two_workers.json',baseline)
    directory=output/'third_worker';directory.mkdir();gate=output/'start.gate'
    cmd=[PYTHON,'-u',str(RUN/'scripts/worker.py'),'--phase','train','--output',str(directory),
        '--chunk','100','--seed','2','--epochs','100','--gate',str(gate)]
    with (directory/'stdout.log').open('w') as out,(directory/'stderr.log').open('w') as err:
        proc=subprocess.Popen(cmd,cwd=str(ROOT),env=worker_environment(),stdout=out,stderr=err,start_new_session=True)
    write_json(output/'third_worker_job.json',dict(pid=proc.pid,command=cmd,started_at=now(),formal_model=False))
    deadline=time.monotonic()+300
    while not (directory/'ready.txt').exists():
        assert proc.poll() is None,'Third worker failed before ready'
        assert time.monotonic()<deadline
        time.sleep(1)
    gate.write_text(now())
    write_json(RUN/'reports/live_parallel_tuning.json',dict(status='measuring_three_workers',directory=str(output),third_pid=proc.pid))
    while proc.poll() is None:
        sample(output/'resources.jsonl','three_workers');time.sleep(10)
    result=dict(status='completed',completed_at=now(),baseline=baseline,directory=str(output),
                third_worker_exit_code=proc.returncode,formal_model_directories={k:str(v) for k,v in active.items()},
                note='Live workload contains k100 and k1; third k100 worker is a separate 100-epoch test, excluded from formal results')
    if proc.returncode==0:
        series=rows(directory/'epoch_metrics.jsonl')
        assert len(series)==100
        start=series[5]['epoch_start_monotonic'];end=series[-1]['epoch_end_monotonic']
        counts={k:completed(p,start,end) for k,p in active.items()}
        counts['third_test']=completed(directory,start,end)
        trial=dict(window_seconds=end-start,workers=counts,total_epochs_per_second=sum(counts.values())/(end-start))
        result.update(three_workers=trial,three_vs_two_ratio=trial['total_epochs_per_second']/baseline['total_epochs_per_second'])
        result['selected_train_parallel']=3 if result['three_vs_two_ratio']>1.02 else 2
    else:
        result.update(selected_train_parallel=2,error=(directory/'stderr.log').read_text(errors='replace')[-4000:])
    write_json(output/'result.json',result)
    write_json(RUN/'reports/live_parallel_tuning.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
