"""Paired inference sensitivity to software rasterizers, never formal rollouts."""
import sys
import json
import pickle
from common import ROOT,RUN,official_args,install_official_argv,activate_renderer,write_json
sys.path.insert(0,str(ROOT))


def main():
    model=RUN/'tests/training_throughput_1/seed0'
    install_official_argv(official_args(model,100,0,30))
    activate_renderer()
    import numpy as np
    import torch
    import h5py
    import imitate_episodes as official
    torch.set_num_threads(1)
    official.set_seed(1000)
    config=dict(lr=1e-5,num_queries=100,kl_weight=10,hidden_dim=512,dim_feedforward=3200,
        lr_backbone=1e-5,backbone='resnet18',enc_layers=4,dec_layers=7,nheads=8,camera_names=['top'])
    policy=official.make_policy('ACT',config)
    policy.load_state_dict(torch.load(str(model/'policy_best.ckpt'),map_location='cpu'))
    policy.cuda().eval()
    with (model/'dataset_stats.pkl').open('rb') as f:stats=pickle.load(f)
    with h5py.File(str(ROOT/'data/sim_transfer_cube_scripted/episode_0.hdf5'),'r') as f:qpos=f['observations/qpos'][:6]
    files={name:np.load(str(RUN/'tests'/('render_'+name)/'comparison.npz')) for name in ['softpipe','llvmpipe']}
    outputs={}
    with torch.inference_mode():
        for name,data in files.items():
            predictions=[]
            for i,image in enumerate(data['images']):
                state=(qpos[i%6]-stats['qpos_mean'])/stats['qpos_std']
                state=torch.from_numpy(state).float().cuda()[None]
                pixels=torch.from_numpy(image.transpose(2,0,1).copy()/255.).float().cuda()[None,None]
                predictions.append(policy(state,pixels).cpu().numpy())
            outputs[name]=np.concatenate(predictions)
    delta=np.abs(outputs['softpipe']-outputs['llvmpipe'])
    raw_delta=delta*stats['action_std'][None,None,:]
    result=dict(samples=12,model='Separate 30-epoch preliminary throughput model; not a formal V2 model',
        normalized_action_mae=float(delta.mean()),normalized_action_max_abs_difference=float(delta.max()),
        denormalized_action_mae=float(raw_delta.mean()),per_action_dimension_mae=raw_delta.mean(axis=(0,1)).tolist(),
        interpretation='Same proprioceptive inputs, paired old/new rasterized images; small-sample open-loop prediction sensitivity only, not a policy success-rate comparison',
        finite=bool(np.isfinite(delta).all()))
    assert result['finite']
    write_json(RUN/'reports/software_renderer_policy_sensitivity.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
