from dm_control import suite
import numpy as np

domain_name = 'walker'
task_name = 'walk'
env = suite.load(domain_name, task_name)

action_spec = env.action_spec()
time_step = env.reset()

action = np.random.uniform(action_spec.minimum,
                           action_spec.maximum,
                           size=action_spec.shape)
time_step = env.step(action)
print(time_step.last(), time_step.reward, time_step.discount, time_step.observation)