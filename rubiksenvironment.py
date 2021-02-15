import gym
from gym import spaces
from RubixCube import RubiksCube 
from tf_agents.specs import tensor_spec, array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np

class RubiksEnvironment(gym.Env):

	def __init__(self, num_cubes):
		super(RubixEnv, self).__init__()
		num_faces = 6
		num_dims, num_layers, num_directions = 3, num_cubes, 2
		self._action_space = array_spec.BoundedArraySpec((num_dims * num_layers * num_directions, ), np.int32)
		self._action_spec = tensor_spec.BoundedTensorSpec((), np.int32, minimum=0, maximum=8, name='action')
		self._observation_spec = tensor_spec.BoundedTensorSpec((num_faces, num_cubes, num_cubes), np.int32, minimum=0, maximum=num_cubes - 1)
		self._time_step_spec = ts.time_step_spec(self._observation_spec)
		self._state = RubiksCube().random_position(10)
		self.seed = 1

	def step(self, action):
		cube = RubixCube().load_env(self.state)
		cube.move(action)
		obs = cube.int_conversion()
		reward = self.reward()
		is_done = reward == 0

		return obs, reward, is_done, {}	

	def reward(self):
		counts = collections.Counter(self.state)
		total_counts = float(sum(counts))
		probs = [float(count)/total_counts for count in counts]
		entropy = -1.0* sum([prob * np.log(prob) for prob in probs])

		return entropy

	def reset(self):
		state = RubiksCube().random_position(10, self.seed)

		return state

	def action_spec(self):
		return self._action_spec

	def observation_spec(self):
		return self._observation_spec

	def time_step_spec(self):
		return self._time_step_spec



