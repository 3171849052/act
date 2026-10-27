import json
import sys
import time
from common import ROOT,RUN,write_json,now
sys.path.insert(0,str(ROOT))

def main():
    import numpy as np
    import h5py
    import sim_env
    from render_controls import install_top_only,top_only_source
    original=sim_env.BimanualViperXTask.get_observation
    top_only_source() # AST audit before any test.
    with h5py.File(ROOT/'data/sim_transfer_cube_scripted/episode_0.hdf5','r') as f:
        actions=f['action'][:8]
    poses=[[0.03,0.42,0.05,1,0,0,0],[0.17,0.58,0.05,1,0,0,0]]
    reference=[]; result={'checked_at':now(),'steps_per_pose':8,'poses':poses,'comparisons':[]}
    for optimized in [False,True]:
        sim_env.BimanualViperXTask.get_observation=original
        if optimized: install_top_only()
        elapsed=[]
        for case,pose in enumerate(poses):
            np.random.seed(1000)
            sim_env.BOX_POSE[0]=np.array(pose,dtype=float)
            env=sim_env.make_sim_env('sim_transfer_cube_scripted')
            start=time.perf_counter(); ts=env.reset(); trajectory=[]
            for i in range(9):
                if i: ts=env.step(actions[i-1])
                obs=ts.observation
                item={'image':obs['images']['top'].copy(),'qpos':obs['qpos'].copy(),
                      'qvel':obs['qvel'].copy(),'state':env.physics.get_state().copy(),'reward':ts.reward,
                      'numpy_rng':np.random.get_state()}
                trajectory.append(item)
            elapsed.append(time.perf_counter()-start)
            env.close()
            if not optimized:
                reference.append(trajectory)
            else:
                for step,(a,b) in enumerate(zip(reference[case],trajectory)):
                    for key in ['image','qpos','qvel','state']:
                        assert np.array_equal(a[key],b[key]),(case,step,key)
                    assert a['reward']==b['reward']
                    assert a['numpy_rng'][0]==b['numpy_rng'][0]
                    assert np.array_equal(a['numpy_rng'][1],b['numpy_rng'][1])
                    assert a['numpy_rng'][2:]==b['numpy_rng'][2:]
                result['comparisons'].append({'pose_index':case,'observations':len(trajectory),
                                              'pixels_states_rewards_rng_bitwise_equal':True})
            print('camera_test',optimized,'pose',case,'seconds',elapsed[-1],flush=True)
        result['top_only_seconds' if optimized else 'original_seconds']=sum(elapsed)
    result.update(passed=True,measured_speedup=result['original_seconds']/result['top_only_seconds'])
    write_json(RUN/'reports/camera_filter_validation.json',result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
