"""
The main DDPG test oin minecraft simple.
Author: William Guss
"""
import gym
import tensorflow as tf
import gym_minecraft
import numpy as np

def main():
	"""
	Runs a simple gym miencraft test.
	"""
	env = gym.make('MinecraftBasic-v0')
	env.configure(
		start_minecraft=False,
		allowDiscreteMovement=False, 
		allowContinuousMovement=True, 
		continuous_discrete=False,
		mission_file='./simple_mission.xml')
	cts_action_space, mdesc_action_space = env.action_spaces

	# Set up tensorflow session
	sess = tf.InteractiveSession()

	env.reset()

	done = False
	while not done:
		state, reward, done, info = env.step([cts_action_space.sample(), [0,0,0,0]])
		print(reward)



if __name__ == '__main__':
	main()
