"""Extract verified observations only; fail closed on incomplete experiments."""
import csv
import json
from pathlib import Path
from common import RUN, MODELS, write_json, validate_artifacts, parse_evaluation

def write_csv(path,rows):
    assert rows
    with path.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def collect():
    state = json.loads((RUN/'reports/run_state.json').read_text())
    preflight = json.loads((RUN/'reports/preflight.json').read_text())
    summaries,episodes,manifest = [],[],[]
    stats_hashes = set()
    for name,policy,chunk in MODELS:
        training = state['models'][name]
        assert training['status']=='SUCCEEDED'
        directory = Path(training['directory'])
        artifacts = validate_artifacts(directory,1000)
        stats_hashes.add(artifacts['stats_sha256'])
        split = json.loads((directory/'data_split.json').read_text())
        for key in ['train','validation']: assert split[key]==preflight['split'][key]
        evaluation = state['evaluations'][name]
        assert evaluation['status']=='SUCCEEDED' and evaluation['exit_code']==0
        eval_dir = Path(evaluation['directory'])
        for filename in ['policy_best.ckpt','dataset_stats.pkl']:
            assert (eval_dir/filename).resolve()==(directory/filename).resolve()
        metrics = parse_evaluation(eval_dir)
        write_json(eval_dir/'evaluation_metrics.json',metrics)
        attempts = training['attempts']
        successful = attempts[-1]
        row = dict(model=name,policy_class=policy,chunk_size=chunk if policy=='ACT' else '',
                   training_epochs=1000,training_seed=0,training_status=training['status'],
                   training_seconds=successful['duration_seconds'],
                   training_total_attempt_seconds=sum(a['duration_seconds'] for a in attempts),
                   evaluation_status=evaluation['status'],evaluation_seconds=evaluation['duration_seconds'],
                   successes=metrics['successes'],episodes=metrics['episodes'],
                   success_rate=metrics['success_rate'],average_return=metrics['average_return'])
        row.update({'reward_ge_'+str(r)+'_rate':metrics['threshold_rates'][str(r)] for r in range(5)})
        summaries.append(row)
        for i,(ret,reward) in enumerate(zip(metrics['returns'],metrics['highest_rewards'])):
            episodes.append(dict(model=name,episode_index=i,episode_return=ret,
                                 highest_reward=reward,success=int(reward==4),
                                 video=str(eval_dir/f'video{i}.mp4') if i<2 else ''))
        manifest.append(dict(model=name,policy_class=policy,chunk_size=chunk if policy=='ACT' else '',
                             dataset='sim_transfer_cube_scripted',episodes_in_dataset=50,
                             batch_size=8,lr=1e-5,num_epochs=1000,seed=0,
                             kl_weight=10 if policy=='ACT' else '',hidden_dim=512 if policy=='ACT' else '',
                             dim_feedforward=3200 if policy=='ACT' else '',temporal_aggregation=False,
                             model_directory=str(directory),evaluation_directory=str(eval_dir),
                             checkpoint=str(directory/'policy_best.ckpt'),stats=str(directory/'dataset_stats.pkl'),
                             checkpoint_sha256=artifacts['checkpoint_sha256'],stats_sha256=artifacts['stats_sha256'],
                             training_config=str(directory/'training_config.json'),
                             evaluation_config=str(eval_dir/'evaluation_config.json'),
                             render_backend=evaluation.get('render_backend','osmesa'),
                             top_camera_only=evaluation.get('top_camera_only',False),
                             training_stdout=successful['stdout'],training_stderr=successful['stderr'],
                             evaluation_stdout=evaluation['stdout'],evaluation_stderr=evaluation['stderr'],
                             start_time=successful['start_time'],end_time=successful['end_time'],
                             exit_code=successful['exit_code']))
    assert len(stats_hashes)==1, 'Official normalization stats must match for the same dataset'
    write_csv(RUN/'reports/summary.csv',summaries)
    write_csv(RUN/'reports/episode_level_results.csv',episodes)
    write_csv(RUN/'reports/experiment_manifest.csv',manifest)
    print('Collected four models and 120 verified evaluation episodes',flush=True)
    return summaries

if __name__=='__main__': collect()
