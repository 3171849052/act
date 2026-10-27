"""Verify prepared data after machine failure, then enable and run all V3 tasks."""
import json
import subprocess
import time
from pathlib import Path
from common import ROOT, RUN, PYTHON, sha256, now, write_json, worker_environment


def main():
    report = RUN / 'reports/recovery_data_verification.json'
    result = dict(started_at=now(), status='VERIFYING', files_verified=0)
    write_json(report, result)
    try:
        prepared = json.loads((RUN / 'reports/insertion_data_preparation/preparation_results.json').read_text())
        if prepared.get('status') != 'COMPLETED':
            subprocess.run([PYTHON, '-u', str(RUN / 'scripts/prepare_insertion_data.py')],
                           cwd=str(ROOT), env=worker_environment(), check=True)
            prepared = json.loads((RUN / 'reports/insertion_data_preparation/preparation_results.json').read_text())
        assert prepared['status'] == 'COMPLETED' and prepared['archive_bytes_preserved']
        for task, dataset in prepared['datasets'].items():
            assert dataset['ready'] and len(dataset['files']) == 50
            assert {f['id'] for f in dataset['files']} == set(range(50))
            for item in dataset['files']:
                path = Path(item['path'])
                assert path.stat().st_size == item['bytes'] and sha256(path) == item['sha256'], str(path)
                result['files_verified'] += 1
                if result['files_verified'] % 10 == 0:
                    write_json(report, result)
                    print('Verified', result['files_verified'], '/ 100', flush=True)
        plan_path = RUN / 'experiment_plan.json'
        plan = json.loads(plan_path.read_text())
        for spec in plan['tasks']:
            validation = RUN / 'reports/data' / spec['name'] / 'validation.json'
            assert json.loads(validation.read_text())['passed']
            spec['enabled'] = True
            spec.pop('hold_reason', None)
        plan['execution'].update(overlap_training_processes=2, after_v2_training_processes=2,
                                 active_scope='All three V3 tasks; user authorized Insertion training after data verification',
                                 recovery_authorization='Restart interrupted V2/V3 and launch Insertion experiments, maximize safe parallelism')
        write_json(plan_path, plan)
        state_path = RUN / 'reports/run_state.json'
        state = json.loads(state_path.read_text())
        for group in ['models', 'evaluations']:
            for spec in plan['tasks']:
                rec = state[group].setdefault(spec['name'], dict(status='PENDING'))
                if rec['status'] in ['WAITING_FOR_USER_DATA', 'BLOCKED_DATA']:
                    rec['status'] = 'PENDING'
                    rec.pop('reason', None)
        write_json(state_path, state)
        result.update(status='PASSED', completed_at=now(), training_processes=2)
        write_json(report, result)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        with (RUN/'logs'/('recovery_queue_'+stamp+'.stdout.log')).open('x') as out, (RUN/'logs'/('recovery_queue_'+stamp+'.stderr.log')).open('x') as err:
            child = subprocess.Popen([PYTHON, '-u', str(RUN/'scripts/wait_for_v2_then_run.py')],
                cwd=str(ROOT), env=worker_environment(), stdout=out, stderr=err,
                stdin=subprocess.DEVNULL, start_new_session=True)
        result['queue_pid'] = child.pid
        write_json(report, result)
        print('V3 persistent queue started:', child.pid, flush=True)
    except BaseException as exc:
        result.update(status='FAILED', error=repr(exc), ended_at=now())
        write_json(report, result)
        raise


if __name__ == '__main__':
    main()
