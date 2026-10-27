"""Official ACT execution with V2-only observability and video selection hooks."""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
from common import (ROOT, RUN, now, write_json, sha256, check_sources, official_args,
                    install_official_argv, activate_renderer, validate_model)
sys.path.insert(0, str(ROOT))


def insert_checked(source, edits):
    result = source
    for old, new in edits:
        assert result.count(old) == 1, old
        result = result.replace(old, new)
    restored = result
    for old, new in reversed(edits):
        assert restored.count(new) == 1
        restored = restored.replace(new, old)
    assert ast.dump(ast.parse(restored)) == ast.dump(ast.parse(source))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['train','eval'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk', type=int, choices=[1,100,400], required=True)
    parser.add_argument('--seed', type=int, choices=[0,1,2], required=True)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--temporal', action='store_true')
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--smoke_episodes', type=int)
    a = parser.parse_args()
    assert a.output.is_dir()
    if a.smoke_episodes is not None:
        assert RUN/'tests' in a.output.parents and 1 <= a.smoke_episodes <= 2
    args = official_args(a.output,a.chunk,a.seed,a.epochs,a.temporal,a.phase=='eval')
    install_official_argv(args)  # DETR's constructor reparses this argv.
    check_sources()
    activate_renderer()  # Must precede MuJoCo, PyOpenGL, and the official module.
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import imitate_episodes as official
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # No AMP, TF32 override, loss/optimizer changes, or checkpoint initialization.
    record = dict(args, python=sys.executable, git_commit=json.loads((RUN/'reports/preflight.json').read_text())['git_commit'],
        upstream_commit=json.loads((RUN/'provenance/official_comparison.json').read_text())['commit'],
        source_manifest=sha256(RUN/'provenance/source_manifest.json'), split_seed=1,
        camera_names=['top'], backbone='resnet18', torch_threads=1, amp=False,
        backend=os.environ['MUJOCO_GL'], renderer=os.environ.get('ACT_V2_RENDERER'),
        environment={key:os.environ.get(key) for key in ['MUJOCO_GL','PYOPENGL_PLATFORM','LD_LIBRARY_PATH',
                     'GALLIUM_DRIVER','LP_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
        initialization=dict(e1_checkpoint_loaded=False, policy='fresh official ACT initialization',
                            backbone='official ImageNet-pretrained ResNet18'), started_at=now())
    pretrained=Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth'
    if pretrained.exists():
        record['pretrained_backbone_sha256']=sha256(pretrained)
    if a.phase == 'train':
        config_path=a.output/'training_config.json'
        assert not config_path.exists(), 'Never overwrite or resume a partial training run'
        write_json(config_path,record)
        original_loader=official.load_data
        def checked_loader(*xs,**kwargs):
            train,val,stats,is_sim=original_loader(*xs,**kwargs)
            split=dict(train=train.dataset.episode_ids.tolist(),validation=val.dataset.episode_ids.tolist())
            assert split==json.loads((RUN/'reports/preflight.json').read_text())['split']
            write_json(a.output/'data_split.json',split)
            return train,val,stats,is_sim
        official.load_data=checked_loader
        original_make=official.make_policy
        def fresh_policy(*xs,**kwargs):
            policy=original_make(*xs,**kwargs)
            parameter=policy.state_dict()['model.query_embed.weight'].detach().cpu().numpy()
            record['initialization']['query_embedding_sha256']=hashlib.sha256(parameter.tobytes()).hexdigest()
            record['parameter_count']=sum(p.numel() for p in policy.parameters())
            write_json(config_path,record)
            return policy
        official.make_policy=fresh_policy
        epoch_start=[None]
        def begin(epoch):
            torch.cuda.synchronize()
            epoch_start[0]=time.perf_counter()
        def end(epoch,train,val,best):
            torch.cuda.synchronize()
            end_time=time.perf_counter()
            row=dict(epoch=epoch,time=now(),epoch_seconds=end_time-epoch_start[0],
                epoch_start_monotonic=epoch_start[0],epoch_end_monotonic=end_time,
                train={k:float(v.item()) for k,v in train.items()},
                validation={k:float(v.item()) for k,v in val.items()},best_epoch=int(best[0]),
                best_validation_loss=float(best[1].item()),gpu_peak_memory_bytes=torch.cuda.max_memory_allocated())
            assert np.isfinite(list(row['train'].values())+list(row['validation'].values())).all()
            with (a.output/'epoch_metrics.jsonl').open('a') as f:
                f.write(json.dumps(row,allow_nan=False)+'\n')
                f.flush()
            write_json(a.output/'progress.json',dict(epochs_completed=epoch+1,target_epochs=a.epochs,
                updated_at=now(),last_epoch_seconds=row['epoch_seconds'],best_epoch=row['best_epoch'],
                best_validation_loss=row['best_validation_loss']))
        source=inspect.getsource(official.train_bc)
        instrumented=insert_checked(source,[
            ('    for epoch in tqdm(range(num_epochs)):\n',
             '    for epoch in tqdm(range(num_epochs)):\n        _v2_epoch_start(epoch)\n'),
            ("    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')\n",
             "        _v2_epoch_end(epoch, epoch_summary, validation_history[-1], best_ckpt_info)\n\n    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')\n")])
        (a.output/'training_loop.original.py').write_text(source)
        (a.output/'training_loop.instrumented.py').write_text(instrumented)
        official._v2_epoch_start=begin; official._v2_epoch_end=end
        exec(compile(instrumented,str(a.output/'training_loop.instrumented.py'),'exec'),official.__dict__)
        original_train=official.train_bc
        def gated_train(*xs,**kwargs):
            if a.gate:
                (a.output/'ready.txt').write_text(now())
                deadline=time.monotonic()+300
                while not a.gate.exists():
                    assert time.monotonic()<deadline,'Benchmark gate timed out'
                    time.sleep(.1)
            return original_train(*xs,**kwargs)
        official.train_bc=gated_train
        original_plot=official.plot_history
        def closed_plot(*xs,**kwargs):
            original_plot(*xs,**kwargs)
            plt.close('all')
        official.plot_history=closed_plot
        official.main(args)  # split seed 1; train_bc subsequently sets training seed 0/1/2.
        artifacts=validate_model(a.output,a.epochs)
        write_json(a.output/'artifact_validation.json',artifacts)
        write_json(a.output/'worker_completion.json',dict(completed=True,ended_at=now(),artifacts=artifacts))
    else:
        from render_controls import install_top_only, identity
        selection=json.loads((RUN/'reports/render_selection.json').read_text())
        assert selection['camera_filter_validated']
        install_top_only(a.output)
        config_path=a.output/'evaluation_config.json'
        assert not config_path.exists()
        record.update(eval_seed=1000,episodes=a.smoke_episodes or 50,steps_per_episode=400,
            temporal_aggregation=a.temporal,video_selection='First actual success and first actual failure, at most two; absence of a class recorded',
            checkpoint_sha256=sha256(a.output/'policy_best.ckpt'),stats_sha256=sha256(a.output/'dataset_stats.pkl'))
        write_json(config_path,record)
        starts={}; episodes=[]; videos={}; gl_info={}; physical_stages={}
        import sim_env
        original_reward=sim_env.TransferCubeTask.get_reward
        geom_cache={}
        def observed_reward(task,physics):
            reward=original_reward(task,physics)
            # Read-only contact evidence; preserve the original reward return value.
            key=id(physics)
            if key not in geom_cache:
                geom_cache[key]={name:physics.model.name2id(name,'geom') for name in [
                    'red_box','table','vx300s_left/10_left_gripper_finger','vx300s_right/10_right_gripper_finger']}
            ids=geom_cache[key]
            pairs={frozenset((int(physics.data.contact[i].geom1),int(physics.data.contact[i].geom2)))
                   for i in range(int(physics.data.ncon))}
            left=frozenset((ids['red_box'],ids['vx300s_left/10_left_gripper_finger'])) in pairs
            right=frozenset((ids['red_box'],ids['vx300s_right/10_right_gripper_finger'])) in pairs
            table=frozenset((ids['red_box'],ids['table'])) in pairs
            physical_stages['touch_cube']=physical_stages.get('touch_cube',False) or left or right
            physical_stages['lift_cube']=physical_stages.get('lift_cube',False) or ((left or right) and not table)
            physical_stages['left_contact']=physical_stages.get('left_contact',False) or left
            return reward
        sim_env.TransferCubeTask.get_reward=observed_reward
        def episode_start(index):
            starts[index]=time.perf_counter()
            physical_stages.clear()
        def env_identity(env):
            if not gl_info:
                gl_info.update(identity(env.physics))
                if selection['renderer']=='llvmpipe':
                    assert 'llvmpipe' in gl_info['GL_RENDERER'].lower()
                if selection['renderer']=='egl':
                    assert 'NVIDIA' in gl_info['GL_VENDOR']
                write_json(a.output/'gl_identity.json',gl_info)
        def episode_done(index,rewards,total,highest,pose,qpos,actions,env):
            rewards=np.asarray(rewards,dtype=float)
            assert rewards.shape==(400,) and np.isfinite(rewards).all()
            assert np.isfinite(qpos).all() and np.isfinite(actions).all()
            assert np.isfinite(env.physics.get_state()).all()
            assert not np.any(env.physics.data.warning.number)
            assert identity(env.physics)['gl_error']==0
            row=dict(episode=index,steps=400,initial_box_pose=np.array(pose).tolist(),
                episode_return=float(total),highest_reward=int(highest),success=bool(highest==4),
                reward_ge_1=bool(highest>=1),reward_ge_2=bool(highest>=2),reward_ge_3=bool(highest>=3),
                elapsed_seconds=time.perf_counter()-starts[index],rewards=rewards.tolist(),
                physical_stages=dict(physical_stages))
            with (a.output/'episodes.jsonl').open('a') as f:
                f.write(json.dumps(row,allow_nan=False)+'\n')
            np.savez_compressed(str(a.output/('episode_%02d.npz'%index)),
                rewards=rewards,qpos=np.array(qpos),actions=np.array(actions),initial_box_pose=np.array(pose))
            episodes.append(row)
            write_json(a.output/'progress.json',dict(episodes_completed=len(episodes),target_episodes=record['episodes'],
                successes=sum(e['success'] for e in episodes),updated_at=now()))
        def select_video(success,index):
            label='success' if success else 'failure'
            if label in videos:
                return False
            videos[label]=dict(episode=index,path='video%d.mp4'%index,selection='first '+label)
            return True
        source=inspect.getsource(official.eval_bc)
        edits=[('    for rollout_id in range(num_rollouts):\n',
                '    for rollout_id in range(num_rollouts):\n        _v2_episode_start(rollout_id)\n'),
               ('        ts = env.reset()\n','        ts = env.reset()\n        _v2_env_identity(env)\n'),
               ('        highest_rewards.append(episode_highest_reward)\n',
                '        highest_rewards.append(episode_highest_reward)\n        _v2_episode_done(rollout_id, rewards, episode_return, episode_highest_reward, BOX_POSE[0], qpos_list, target_qpos_list, env)\n'),
               ('        if save_episode:\n',
                '        if save_episode and _v2_select_video(episode_highest_reward == env_max_reward, rollout_id):\n')]
        if a.smoke_episodes:
            edits.append(('    num_rollouts = 50\n','    num_rollouts = %d\n'%a.smoke_episodes))
        edits.append(('    return success_rate, avg_return\n',
                      '    env.physics.free()\n    return success_rate, avg_return\n'))
        instrumented=insert_checked(source,edits)
        (a.output/'evaluation.original.py').write_text(source)
        (a.output/'evaluation.instrumented.py').write_text(instrumented)
        official._v2_episode_start=episode_start; official._v2_episode_done=episode_done
        official._v2_select_video=select_video; official._v2_env_identity=env_identity
        exec(compile(instrumented,str(a.output/'evaluation.instrumented.py'),'exec'),official.__dict__)
        policy_config=dict(lr=1e-5,num_queries=a.chunk,kl_weight=10,hidden_dim=512,dim_feedforward=3200,
            lr_backbone=1e-5,backbone='resnet18',enc_layers=4,dec_layers=7,nheads=8,camera_names=['top'])
        config=dict(args,episode_len=400,state_dim=14,policy_config=policy_config,camera_names=['top'],real_robot=False)
        official.set_seed(1)
        rate,avg=official.eval_bc(config,'policy_best.ckpt',save_episode=True)
        assert len(episodes)==record['episodes']
        assert rate==sum(e['success'] for e in episodes)/len(episodes)
        for v in videos.values():
            assert (a.output/v['path']).stat().st_size>1000
        write_json(a.output/'videos.json',dict(selected=videos,
            missing_classes=[k for k in ['success','failure'] if k not in videos],
            note='All 50 episodes determine success rate; videos are illustrative only'))
        write_json(a.output/'evaluation_metrics.json',dict(episodes=len(episodes),successes=sum(e['success'] for e in episodes),
            success_rate=float(rate),average_return=float(avg),
            threshold_rates={str(k):sum(e['highest_reward']>=k for e in episodes)/len(episodes) for k in range(1,5)},
            renderer=gl_info,context_released=True))
        write_json(a.output/'worker_completion.json',dict(completed=True,ended_at=now(),episodes=len(episodes)))


if __name__=='__main__':
    main()
