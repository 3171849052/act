"""Replace only the owned supervisor; keep live training/optimizer processes."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from common import ROOT,RUN,PYTHON,now,write_json,worker_environment,sha256


def process(pid,script=None,directory=None):
    root=Path('/proc/%d'%pid)
    assert root.stat().st_uid==os.getuid()
    fields=(root/'stat').read_text().rsplit(')',1)[1].split()
    command=(root/'cmdline').read_bytes().split(b'\0')
    if script:assert str(script).encode() in command
    if directory:assert str(directory).encode() in command
    return dict(pid=pid,start_ticks=fields[19],state=fields[0],command=[s.decode() for s in command if s])


def main():
    result=json.loads((RUN/'reports/live_parallel_tuning.json').read_text())
    assert result['status']=='completed','Wait for the uncontended live comparison to finish'
    state=json.loads((RUN/'reports/run_state.json').read_text())
    assert state['phase']=='formal_training' and state['status']=='RUNNING'
    old_pid=state['pid'];supervisor=process(old_pid,RUN/'scripts/pipeline.py')
    workers={name:process(rec['pid'],RUN/'scripts/worker.py',rec['directory'])
             for name,rec in state['models'].items() if rec['status']=='RUNNING'}
    assert len(workers)==2
    controls=dict(train_parallel=result['selected_train_parallel'],eval_parallel=state['eval_parallel'],
        evaluations_during_training=1,updated_at=now(),
        reason='User requested more parallelism; live 100-epoch third-worker test plus early evaluation overlap',
        tuning_report='reports/live_parallel_tuning.json')
    write_json(RUN/'parallel_controls.json',controls)
    backup=RUN/'backups'/('scheduler_handoff_'+time.strftime('%Y%m%d_%H%M%S'))
    backup.mkdir()
    # Freeze state-file updates only, never the workers or their process groups.
    os.kill(old_pid,signal.SIGSTOP)
    try:
        for _ in range(100):
            if process(old_pid,RUN/'scripts/pipeline.py')['state']=='T':break
            time.sleep(.02)
        else:raise RuntimeError('Supervisor did not stop')
        stable=json.loads((RUN/'reports/run_state.json').read_text())
        assert stable['pid']==old_pid and stable['phase']=='formal_training'
        for name,rec in stable['models'].items():
            if rec['status']=='RUNNING':
                assert rec['pid']==workers[name]['pid']
                assert process(rec['pid'],RUN/'scripts/worker.py',rec['directory'])['start_ticks']==workers[name]['start_ticks']
        shutil.copy2(str(RUN/'reports/run_state.json'),str(backup/'run_state.before.json'))
        shutil.copy2(str(RUN/'status.txt'),str(backup/'status.before.txt'))
        write_json(backup/'processes.before.json',dict(supervisor=supervisor,workers=workers))
        # SIGTERM targets the supervisor PID, not its session or worker groups.
        os.kill(old_pid,signal.SIGTERM)
        os.kill(old_pid,signal.SIGCONT)
    except BaseException:
        try:os.kill(old_pid,signal.SIGCONT)
        except ProcessLookupError:pass
        raise
    deadline=time.monotonic()+15
    while Path('/proc/%d/stat'%old_pid).exists():
        fields=Path('/proc/%d/stat'%old_pid).read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z':break
        assert time.monotonic()<deadline
        time.sleep(.1)
    stamp=time.strftime('%Y%m%d_%H%M%S')
    stdout=RUN/'logs'/('parallel_pipeline_'+stamp+'.stdout.log')
    stderr=RUN/'logs'/('parallel_pipeline_'+stamp+'.stderr.log')
    with stdout.open('x') as out,stderr.open('x') as err:
        proc=subprocess.Popen([PYTHON,'-u',str(RUN/'scripts/pipeline.py'),'--continue_run','--adopt_running'],
            cwd=str(ROOT),env=worker_environment(),stdout=out,stderr=err,start_new_session=True,stdin=subprocess.DEVNULL)
    deadline=time.monotonic()+45
    while True:
        assert proc.poll() is None,'New supervisor failed; inspect '+str(stderr)
        current=json.loads((RUN/'reports/run_state.json').read_text())
        if current.get('pid')==proc.pid and current['phase']=='overlapped_training_and_evaluation':break
        assert time.monotonic()<deadline
        time.sleep(.2)
    after={name:process(info['pid'],RUN/'scripts/worker.py',current['models'][name]['directory']) for name,info in workers.items()}
    assert all(after[name]['start_ticks']==info['start_ticks'] for name,info in workers.items())
    write_json(RUN/'reports/parallel_scheduler_migration.json',dict(completed_at=now(),old_supervisor=supervisor,
        new_supervisor_pid=proc.pid,workers_before=workers,workers_after=after,
        training_workers_restarted=False,optimizer_state_preserved_in_live_processes=True,
        controls=controls,backup=str(backup),stdout=str(stdout),stderr=str(stderr)))
    print('Supervisor handoff completed; all training worker PIDs and start times preserved.')
    print(json.dumps(controls,indent=2))


if __name__=='__main__':main()
