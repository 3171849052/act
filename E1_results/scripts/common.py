import ast
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
from datetime import datetime, timezone, timedelta

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parent if RUN.name == 'E1_results' else RUN.parent.parent
MODELS = [('M1_ACT_k100','ACT',100),('M2_CNNMLP','CNNMLP',1),
          ('M3_ACT_k1','ACT',1),('M4_ACT_k10','ACT',10)]

def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()

def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))
    os.replace(str(tmp),str(path))

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): digest.update(block)
    return digest.hexdigest()

def validate_artifacts(directory, epochs):
    import torch
    import numpy as np
    directory = Path(directory)
    metrics = json.loads((directory/'training_metrics.json').read_text())
    assert metrics['epochs_completed'] == epochs
    assert len(metrics['validation']) == epochs and len(metrics['train']) == epochs*5
    assert all(math.isfinite(v) for series in ['train','validation'] for row in metrics[series] for v in row.values())
    with (directory/'dataset_stats.pkl').open('rb') as f: stats = pickle.load(f)
    for key in ['qpos_mean','qpos_std','action_mean','action_std']:
        assert stats[key].shape == (14,) and np.isfinite(stats[key]).all()
        if key.endswith('std'): assert (stats[key]>0).all()
    state = torch.load(directory/'policy_best.ckpt',map_location='cpu')
    assert state and all(torch.isfinite(t).all().item() for t in state.values())
    assert (directory/'policy_last.ckpt').stat().st_size>1000000
    assert (directory/'train_val_loss_seed_0.png').stat().st_size>1000
    config = json.loads((directory/'training_config.json').read_text())
    if config['policy_class']=='ACT':
        assert state['model.query_embed.weight'].shape==(config['chunk_size'],512)
    return {'checkpoint_sha256':sha256(directory/'policy_best.ckpt'),
            'stats_sha256':sha256(directory/'dataset_stats.pkl'),
            'epochs_completed':epochs}

def parse_evaluation(directory):
    import numpy as np
    directory = Path(directory)
    raw = (directory/'result_policy_best.txt').read_text()
    arrays = re.findall(r'^\[[^\n]+\]$',raw,re.M)
    assert len(arrays)==2
    returns,highest = [ast.literal_eval(a) for a in arrays]
    assert len(returns)==len(highest)==30
    assert np.isfinite(returns).all() and np.isfinite(highest).all()
    assert all(v in range(5) for v in highest)
    successes = sum(v==4 for v in highest)
    rate = float(re.search(r'Success rate: ([^\n]+)',raw).group(1))
    avg = float(re.search(r'Average return: ([^\n]+)',raw).group(1))
    assert abs(rate-successes/30)<1e-10 and abs(avg-np.mean(returns))<1e-8
    thresholds = {}
    for r in range(5):
        match = re.search(r'Reward >= '+str(r)+r': (\d+)/(\d+) = ([^%]+)%',raw)
        assert match and int(match[2])==30
        n = sum(v>=r for v in highest)
        assert int(match[1])==n and abs(float(match[3])-n/30*100)<1e-8
        thresholds[str(r)] = n/30
    stdout = (directory/'eval.stdout.log').read_text()
    rollouts = re.findall(r'Rollout (\d+)\nepisode_return=([^,]+), episode_highest_reward=([^,]+), env_max_reward=4, Success: (True|False)',stdout)
    assert [int(x[0]) for x in rollouts]==list(range(30)), 'Need all 30 original rollout log records'
    for i,row in enumerate(rollouts):
        assert float(row[1])==returns[i] and float(row[2])==highest[i]
        assert (row[3]=='True')==(highest[i]==4)
    videos = sorted(directory.glob('video*.mp4'))
    assert [p.name for p in videos]==['video0.mp4','video1.mp4']
    assert all(p.stat().st_size>1000 for p in videos)
    return {'successes':successes,'episodes':30,'success_rate':rate,'average_return':avg,
            'returns':returns,'highest_rewards':highest,'threshold_rates':thresholds,
            'videos':[str(p) for p in videos]}
