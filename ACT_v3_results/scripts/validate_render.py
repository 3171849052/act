"""Validate each enabled task's own OSMesa path and exact top-camera filtering."""
import json
import pickle
import random
import sys
import time
from common import ROOT,RUN,TASKS,activate_renderer,write_json,now
sys.path.insert(0,str(ROOT))


def main():
    from v2_dependency import inspect_v2
    assert inspect_v2()['completed'],'Render validation/evaluation waits for V2 completion'
    activate_renderer()
    import numpy as np
    import torch
    import sim_env
    from PIL import Image
    from utils import sample_box_pose,sample_insertion_pose
    from render_controls import install_top_only,identity
    torch.set_num_threads(1)
    original=sim_env.BimanualViperXTask.get_observation
    report=dict(backend='osmesa',renderer='llvmpipe',validated_at=now(),tasks={})
    for spec in TASKS:
        output=RUN/'tests'/('render_'+spec['name']);output.mkdir(parents=True,exist_ok=True)
        reference=[];timing={};entry=dict(passed=False,task_name=spec['task_name'],episode_len=spec['episode_len'],cases=[])
        for filtered in [False,True]:
            sim_env.BimanualViperXTask.get_observation=original
            if filtered:install_top_only(output)
            elapsed=[]
            for case in range(2):
                np.random.seed(1000+case);torch.manual_seed(1000+case);random.seed(1000+case)
                pose=sample_box_pose() if 'transfer_cube' in spec['task_name'] else np.concatenate(sample_insertion_pose())
                sim_env.BOX_POSE[0]=pose.copy()
                env=sim_env.make_sim_env(spec['task_name'])
                try:
                    warm=env.reset();action=warm.observation['qpos'].copy();action[[6,13]]=np.clip(action[[6,13]],0,1)
                    start=time.perf_counter();ts=env.reset();series=[]
                    for step in range(6):
                        if step:
                            command=action.copy();command[0]+=.001*step;command[7]-=.001*step
                            ts=env.step(command)
                        obs=ts.observation;image=obs['images']['top']
                        assert image.shape==(480,640,3) and image.dtype==np.uint8 and np.ptp(image)>0
                        assert np.isfinite(obs['qpos']).all() and np.isfinite(obs['qvel']).all()
                        if step:assert ts.reward==env.task.get_reward(env.physics)
                        series.append(dict(top=image.copy(),qpos=obs['qpos'].copy(),qvel=obs['qvel'].copy(),
                            env_state=obs['env_state'].copy(),full_state=env.physics.get_state().copy(),reward=ts.reward,
                            numpy_rng=pickle.dumps(np.random.get_state()),python_rng=pickle.dumps(random.getstate()),
                            torch_rng=torch.get_rng_state().numpy().copy()))
                    elapsed.append(time.perf_counter()-start)
                    info=identity(env.physics)
                    assert 'osmesa' in info['context_class'].lower() and 'llvmpipe' in info['GL_RENDERER'].lower()
                    assert not np.any(env.physics.data.warning.number)
                    entry['gl_identity']=info
                finally:env.physics.free()
                if not filtered:
                    reference.append(series)
                    Image.fromarray(series[0]['top']).save(str(output/('case%d_top.png'%case)))
                else:
                    for left,right in zip(reference[case],series):
                        for key in ['top','qpos','qvel','env_state','full_state','torch_rng']:
                            assert np.array_equal(left[key],right[key]),(spec['name'],case,key)
                        for key in ['reward','numpy_rng','python_rng']:assert left[key]==right[key]
                    entry['cases'].append(dict(case=case,initial_pose=pose.tolist(),observations=6,
                        pixels_physics_rewards_numpy_python_torch_rng_exact=True))
                print('Render validation',spec['name'],'filtered',filtered,'case',case,'seconds',elapsed[-1],flush=True)
            timing['top_only' if filtered else 'all_cameras']=elapsed
        entry.update(passed=True,timings_seconds=timing,top_filter_speedup=sum(timing['all_cameras'])/sum(timing['top_only']),context_released=True)
        report['tasks'][spec['task_name']]=entry;write_json(output/'validation.json',entry)
    sim_env.BimanualViperXTask.get_observation=original
    write_json(RUN/'reports/render_selection.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
