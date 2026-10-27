"""Lightweight persistent queue. Training overlap is authorized by latest request."""
import fcntl
import json
import os
import subprocess
import time
from common import RUN,ROOT,PYTHON,now,write_json,worker_environment
from v2_dependency import inspect_v2


def main():
    lock=(RUN/'queue.lock').open('a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
    (RUN/'queue.pid').write_text(str(os.getpid())+'\n')
    started=now();child=None
    while True:
        dependency=inspect_v2()
        write_json(RUN/'reports/v2_dependency_latest.json',dependency)
        plan=json.loads((RUN/'experiment_plan.json').read_text())
        execution=plan['execution']
        permitted=dependency['completed'] or (execution['training_overlap_with_v2_authorized'] and dependency['training_overlap_allowed'])
        ready=execution['implementation_ready']
        status='QUEUED_PREPARING_IMPLEMENTATION' if not ready else ('BLOCKED_BY_V2' if dependency['condition']=='BLOCKED_BY_V2' else 'QUEUED_WAITING_FOR_V2')
        if child:
            code=child.poll()
            status='RUNNING_PIPELINE' if code is None else ('COMPLETED' if (RUN/'finish.txt').exists() and code==0 else 'PIPELINE_EXITED')
        elif ready and permitted:
            write_json(RUN/'reports/dependency_at_pipeline_start.json',dependency)
            stamp=time.strftime('%Y%m%d_%H%M%S')
            command=[PYTHON,'-u',str(RUN/'scripts/pipeline.py')]
            if (RUN/'reports/run_state.json').exists():command.append('--continue_run')
            with (RUN/'logs'/('pipeline_'+stamp+'.stdout.log')).open('x') as out,(RUN/'logs'/('pipeline_'+stamp+'.stderr.log')).open('x') as err:
                child=subprocess.Popen(command,cwd=str(ROOT),env=worker_environment(),
                    stdout=out,stderr=err,start_new_session=True,stdin=subprocess.DEVNULL)
            status='STARTING'
        rec=dict(status=status,pid=os.getpid(),started_at=started,last_check=now(),v2=dependency,
                 pipeline_pid=child.pid if child else None,training_overlap_authorized=execution['training_overlap_with_v2_authorized'])
        write_json(RUN/'reports/queue_state.json',rec)
        if child is None:
            (RUN/'status.txt').write_text(status+'\n更新时间: '+rec['last_check']+'\n'+dependency['reason']+
                '\n已获准在 V2 评估期间先运行一个 V3 训练进程；V3 正式评估仍等待 V2 验收完成。\n')
        with (RUN/'logs/queue_checks.jsonl').open('a') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
        if child and child.poll() is not None:return
        time.sleep(60)


if __name__=='__main__':main()
