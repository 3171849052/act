"""User-authorized camera optimization; model and physics source files are untouched."""
import ast
import inspect
import textwrap
import types
from pathlib import Path

def top_only_source():
    import sim_env
    original = textwrap.dedent(inspect.getsource(sim_env.BimanualViperXTask.get_observation))
    removed = ["    obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')\n",
               "    obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')\n"]
    source = original
    for line in removed:
        assert source.count(line)==1
        source=source.replace(line,'')
    old,new=ast.parse(original),ast.parse(source)
    old.body[0].body=[node for node in old.body[0].body if not (
        isinstance(node,ast.Assign) and any(ast.dump(node)==ast.dump(ast.parse(s.strip()).body[0]) for s in removed))]
    assert ast.dump(old)==ast.dump(new), 'Camera filtering must only remove the two unused renders'
    return original,source

def install_top_only(directory=None):
    import sim_env
    original,source=top_only_source()
    namespace=dict(sim_env.__dict__)
    exec(compile(source,'<verified_top_camera_observation>','exec'),namespace)
    sim_env.BimanualViperXTask.get_observation=namespace['get_observation']
    if directory is not None:
        directory=Path(directory)
        (directory/'observation_source.py').write_text(source)
        (directory/'observation_source.original.py').write_text(original)

def evaluation_source(original):
    count='    num_rollouts = 50\n'
    video='        if save_episode:\n            save_videos('
    assert original.count(count)==original.count(video)==1
    return original.replace(count,'    num_rollouts = config.get("num_eval_episodes", 50)\n').replace(
        video,'        if save_episode and rollout_id < config.get("max_saved_videos", 50):\n            save_videos(')
