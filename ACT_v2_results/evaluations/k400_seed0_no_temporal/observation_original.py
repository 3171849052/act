def get_observation(self, physics):
    obs = collections.OrderedDict()
    obs['qpos'] = self.get_qpos(physics)
    obs['qvel'] = self.get_qvel(physics)
    obs['env_state'] = self.get_env_state(physics)
    obs['images'] = dict()
    obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
    obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
    obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')

    return obs
