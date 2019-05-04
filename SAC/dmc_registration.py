import dm_control2gym
import gym
from dm_control import suite

for task in suite.ALL_TASKS:
    env_id = (task[0] + '_' + task[1]).replace('_', ' ')
    env_id = 'DM' + ''.join(x
                            for x in env_id.title() if not x.isspace()) + '-v2'
    gym.register(
        id=env_id,
        entry_point=dm_control2gym.make,
        kwargs={
            'domain_name': task[0],
            'task_name': task[1]
        },
        max_episode_steps=1000)
