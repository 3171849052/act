"""Detach the scheduler from the initiating tool session, retaining every prior log."""
import os
import subprocess
import sys
import time
from common import RUN, ROOT

lock=RUN/'pipeline.lock'
if lock.exists():
    pid=int(lock.read_text().strip())
    try:
        os.kill(pid,0)
    except ProcessLookupError:
        lock.rename(RUN/('pipeline.lock.stale.'+str(time.time_ns())))
    else:
        raise SystemExit('Scheduler already running: '+str(pid))
logbase='pipeline' if not (RUN/'logs/pipeline.stdout.log').exists() else 'pipeline_'+str(time.time_ns())
with (RUN/'logs'/(logbase+'.stdout.log')).open('x') as out, (RUN/'logs'/(logbase+'.stderr.log')).open('x') as err:
    proc=subprocess.Popen([sys.executable,'-u',str(RUN/'scripts/run_training_jobs.py')],
                          cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                          close_fds=True,start_new_session=True)
(RUN/'launcher.pid').write_text(str(proc.pid)+'\n')
(RUN/'active_pipeline_log.txt').write_text(str(RUN/'logs'/(logbase+'.stdout.log'))+'\n')
print('Detached scheduler PID:',proc.pid,'log:',RUN/'logs'/(logbase+'.stdout.log'),flush=True)
