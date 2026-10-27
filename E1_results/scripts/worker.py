"""Run official training and minimally adapted official evaluation in separate processes."""
import argparse
import json
import sys
import time
import types
from pathlib import Path
from common import ROOT, RUN, write_json, sha256, validate_artifacts

sys.path.insert(0,str(ROOT))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase',choices=['train','eval'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy',choices=['ACT','CNNMLP'],required=True)
    p.add_argument('--chunk',type=int,required=True)
    p.add_argument('--epochs',type=int,default=1000)
    p.add_argument('--num_eval_episodes',type=int,default=50)
    p.add_argument('--max_saved_videos',type=int,default=50)
    p.add_argument('--top_camera_only',action='store_true')
    p.add_argument('--reuse_videos_from',type=Path)
    p.add_argument('--torch_threads',type=int,default=4)
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    args = dict(eval=a.phase=='eval',onscreen_render=False,ckpt_dir=str(a.output),
                policy_class=a.policy,task_name='sim_transfer_cube_scripted',batch_size=8,
                seed=0,num_epochs=a.epochs,lr=1e-5,kl_weight=10,chunk_size=a.chunk,
                hidden_dim=512,dim_feedforward=3200,temporal_agg=False)
    # DETR reparses sys.argv during model creation. Supply only its existing supported options.
    sys.argv = [str(ROOT/'imitate_episodes.py')]
    for k,v in args.items():
        if isinstance(v,bool):
            if v: sys.argv.append('--'+k)
        else: sys.argv += ['--'+k,str(v)]
    import torch
    torch.set_num_threads(a.torch_threads)
    import numpy as np
    import matplotlib.pyplot as plt
    import imitate_episodes as official
    snapshot = RUN/'backups/imitate_episodes.py.original'
    assert sha256(snapshot)==sha256(ROOT/'imitate_episodes.py'), 'Official source changed during experiment'
    record = dict(args,python=sys.executable,official_sha256=sha256(snapshot),
                  split_seed=1,train_ratio=0.8,amp=False,torch_threads=a.torch_threads,
                  num_eval_episodes=a.num_eval_episodes,max_saved_videos=a.max_saved_videos)
    record.update(top_camera_only=a.top_camera_only,
                  reused_video_directory=str(a.reuse_videos_from) if a.reuse_videos_from else None)
    record['source_sha256'] = {name:sha256(ROOT/name) for name in [
        'constants.py','utils.py','policy.py','sim_env.py','detr/main.py',
        'detr/models/detr_vae.py','detr/models/transformer.py','detr/models/backbone.py']}
    weights = Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth'
    if weights.exists(): record['pretrained_resnet18_sha256']=sha256(weights)
    if a.phase=='train':
        assert not (a.output/'training_config.json').exists(), 'Refuse overwriting a training run'
        write_json(a.output/'training_config.json',record)
        original_loader = official.load_data
        def load_data(*xs,**kw):
            train,val,stats,is_sim = original_loader(*xs,**kw)
            split = {'train':train.dataset.episode_ids.tolist(),'validation':val.dataset.episode_ids.tolist()}
            expected = json.loads((RUN/'reports/preflight.json').read_text())['split']
            assert split['train']==expected['train'] and split['validation']==expected['validation']
            write_json(a.output/'data_split.json',split)
            return train,val,stats,is_sim
        official.load_data = load_data
        original_plot = official.plot_history
        def plot_history(train,val,epochs,directory,seed):
            def values(rows): return [{k:float(v.item()) for k,v in row.items()} for row in rows]
            write_json(a.output/'training_metrics.json',dict(epochs_completed=len(val),train=values(train),validation=values(val)))
            original_plot(train,val,epochs,directory,seed)
            plt.close('all')
        official.plot_history = plot_history
        official.main(args)
        write_json(a.output/'artifact_validation.json',validate_artifacts(a.output,a.epochs))
    else:
        assert a.num_eval_episodes>0 and a.max_saved_videos>=0
        if a.top_camera_only:
            validation=json.loads((RUN/'reports/camera_filter_validation.json').read_text())
            assert validation['passed']
            from render_controls import install_top_only
            install_top_only(a.output)
        write_json(a.output/'evaluation_config.json',record)
        source = snapshot.read_text()
        old_count = '    num_rollouts = 50\n'
        old_video = '        if save_episode:\n            save_videos('
        assert source.count(old_count)==source.count(old_video)==1
        source = source.replace(old_count,'    num_rollouts = config.get("num_eval_episodes", 50)\n')
        source = source.replace(old_video,'        if save_episode and rollout_id < config.get("max_saved_videos", 50):\n            save_videos(')
        # Only the rollout count and video-save condition differ from the backed-up source.
        (a.output/'evaluation_source.py').write_text(source)
        module = types.ModuleType('e1_official_evaluation')
        module.__file__ = str(snapshot)
        exec(compile(source,str(snapshot),'exec'),module.__dict__)
        if a.reuse_videos_from:
            import re
            import builtins
            old_log=(a.reuse_videos_from/'eval.stdout.log').read_text()
            pattern=r'Rollout (\d+)\nepisode_return=([^,]+), episode_highest_reward=([^,]+), env_max_reward=4, Success: (True|False)'
            previous={int(m[0]):m[1:] for m in re.findall(pattern,old_log)}
            assert len(previous)>=2
            verified=[]
            def checked_print(*items,**kwargs):
                text=' '.join(str(v) for v in items)
                for match in re.findall(pattern,text):
                    i=int(match[0])
                    if i in previous:
                        assert match[1:]==previous[i], 'Legacy/new rollout metrics disagree: '+str(i)
                        verified.append(i)
                        write_json(a.output/'legacy_prefix_validation.json',
                                   {'expected_indices':sorted(previous),'verified_indices':verified,
                                    'reference_directory':str(a.reuse_videos_from)})
                builtins.print(*items,**kwargs)
            module.print=checked_print
            def reuse_video(images,dt,video_path):
                source_video=a.reuse_videos_from/Path(video_path).name
                assert source_video.stat().st_size>1000
                assert Path(video_path).resolve()==source_video.resolve()
                print('Reused original video after matching rollout metrics:',source_video,flush=True)
            module.save_videos=reuse_video
        policy_config = {'lr':1e-5,'lr_backbone':1e-5,'backbone':'resnet18',
                         'num_queries':a.chunk if a.policy=='ACT' else 1,'camera_names':['top']}
        if a.policy=='ACT':
            policy_config.update(kl_weight=10,hidden_dim=512,dim_feedforward=3200,
                                 enc_layers=4,dec_layers=7,nheads=8)
        config = dict(args,episode_len=400,state_dim=14,policy_config=policy_config,
                      camera_names=['top'],real_robot=False,
                      num_eval_episodes=a.num_eval_episodes,max_saved_videos=a.max_saved_videos)
        module.set_seed(1) # Same main() seed; eval_bc subsequently sets seed 1000, unchanged.
        rate,avg = module.eval_bc(config,'policy_best.ckpt',save_episode=True)
        if a.reuse_videos_from:
            assert sorted(verified)==sorted(previous)
        print('Evaluation completed:',rate,avg,flush=True)
        write_json(a.output/'worker_completion.json',{'completed':True,'episodes':a.num_eval_episodes,
                                                     'success_rate':float(rate),'average_return':float(avg)})

if __name__=='__main__':
    main()
