"""Read-only V2 dependency checks; respects its actual single-seed revised plan."""
import json
import os
from pathlib import Path
from common import V2,now,sha256


def live_processes():
    result=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            if entry.stat().st_uid!=os.getuid():continue
            cmd=(entry/'cmdline').read_bytes().split(b'\0')
            if str(V2/'scripts/worker.py').encode() not in cmd:continue
            if b'--output' not in cmd:continue
            output=Path(cmd[cmd.index(b'--output')+1].decode())
            if V2/'models' not in output.parents and V2/'evaluations' not in output.parents:continue
            if (entry/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':continue
            result.append(dict(pid=int(entry.name),output=str(output)))
        except (FileNotFoundError,PermissionError,ProcessLookupError,ValueError):pass
    return result


def inspect_v2():
    out=dict(checked_at=now(),completed=False,training_overlap_allowed=False,reason='')
    try:
        state=json.loads((V2/'reports/run_state.json').read_text())
        plan=json.loads((V2/'experiment_plan.json').read_text())
        out.update(state_status=state['status'],state_phase=state['phase'],live_workers=live_processes(),
                   expected_models=len(plan['models']),expected_evaluations=len(plan['evaluations']),
                   expected_episodes=len(plan['evaluations'])*plan['evaluation']['episodes'])
        if state['status'] in ['FAILED','BLOCKED','INTERRUPTED'] or state.get('error'):
            out.update(condition='BLOCKED_BY_V2',reason='V2 failed or blocked: '+str(state.get('error',state['status'])));return out
        model_names={m['name'] for m in plan['models']};eval_names={e['name'] for e in plan['evaluations']}
        assert set(state['models'])==model_names and set(state['evaluations'])==eval_names
        trained=all(state['models'][name]['status']=='SUCCEEDED' for name in model_names)
        out['all_v2_models_trained']=trained
        if not (V2/'finish.txt').exists():
            supervisor=Path('/proc/%d/cmdline'%state['pid'])
            alive=supervisor.exists() and str(V2/'scripts/pipeline.py').encode() in supervisor.read_bytes().split(b'\0')
            if state['status']=='RUNNING' and alive:
                out.update(condition='V2_EVALUATING' if trained else 'V2_RUNNING',
                    training_overlap_allowed=trained,reason='V2 running; no finish marker; V3 formal evaluation remains gated')
            else:out.update(condition='BLOCKED_BY_V2',reason='V2 is not completed and its recorded supervisor is not alive')
            return out
        assert state['status']=='COMPLETED' and state['phase']=='completed','Finish marker/state disagreement'
        status_text=(V2/'status.txt').read_text()
        assert 'V2 COMPLETED' in status_text and 'finish.txt: True' in status_text,'V2 text status is not completed'
        assert trained and all(state['evaluations'][name]['status']=='SUCCEEDED' for name in eval_names)
        assert not out['live_workers'],'V2 formal workers still alive'
        integrity=json.loads((V2/'reports/integrity_check.json').read_text())
        assert integrity['passed'] and set(integrity['models'])==model_names
        assert integrity.get('source_unchanged') and integrity.get('matching_initial_state_sequences')
        assert integrity['evaluation_groups']==len(eval_names) and integrity['episodes']==out['expected_episodes']
        finish=(V2/'finish.txt').read_text()
        for name in model_names:
            rec=state['models'][name];art=rec['artifacts'];directory=Path(rec['directory'])
            assert art['epochs']==plan['training']['epochs']==2000
            assert integrity['models'][name]==art
            assert directory.joinpath('policy_best.ckpt').stat().st_size>1000000
            assert directory.joinpath('dataset_stats.pkl').stat().st_size>0
            assert name in finish
        for name in eval_names:
            rec=state['evaluations'][name];assert rec['metrics']['episodes']==plan['evaluation']['episodes']==50
            assert len((Path(rec['directory'])/'episodes.jsonl').read_text().splitlines())==50
            assert name in finish
        latest=max((Path(state['evaluations'][n]['directory'])/'worker_completion.json').stat().st_mtime for n in eval_names)
        assert (V2/'finish.txt').stat().st_mtime>=latest,'Stale V2 finish marker'
        assert (V2/'reports/final_report.md').stat().st_size>1000
        for filename in ['main_success.png','chunk_ablation.png','reward_stage_attainment.png']:
            assert (V2/'figures'/filename).stat().st_size>1000
        out.update(completed=True,training_overlap_allowed=True,condition='V2_VERIFIED_COMPLETED',
            reason='V2 plan, state, completion marker, integrity, selected task artifacts and no-live-worker checks agree',
            evidence_sha256={name:sha256(V2/name) for name in ['finish.txt','experiment_plan.json','reports/integrity_check.json']})
    except Exception as exc:out.update(condition='BLOCKED_BY_V2',reason=repr(exc))
    return out
