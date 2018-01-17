"""
The main DDPG test oin minecraft simple.
Author: William Guss
"""
import gym
import tensorflow as tf
import gym_minecraft
import numpy as np
from agent import DDPG

NUM_EPISODES=1000
NUM_TESTS=10

def test(agent, num_episodes):
	print("Testing")
	env = gym.make('MinecraftBasic-v0')
	env.configure(
		start_minecraft=False,
		allowDiscreteMovement=False, 
		allowContinuousMovement=True, 
		continuous_discrete=False,
		mission_file='./simple_mission_test.xml')
	cts_action_space, mdesc_action_space = env.action_spaces

	for ep in range(num_episodes):
		next_state = env.reset()

		done = False
		while not done:
			state = next_state
			action = agent.action(state)
			cts_act = action[:sum(cts_action_space.shape)]
			desc_act = np.round(action[sum(cts_action_space.shape):]).astype(int).tolist()
			next_state, reward, done, info = env.step([cts_act,desc_act])

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
		mission_file='./simple_mission_train.xml')
	cts_action_space, mdesc_action_space = env.action_spaces

	# Set up tensorflow session
	sess = tf.InteractiveSession()

	agent = DDPG(sess, env.observation_space, env.action_spaces)


	for ep in range(NUM_EPISODES):
		print("EPISODE: {}".format(ep))
		if ep % 25 == 0 and ep > 0:
			test(agent, NUM_TESTS)


		next_state = env.reset()

		done = False
		leng = 0
		while not done:
			state = next_state
			action = agent.noise_action(state)
			cts_act = action[:sum(cts_action_space.shape)]
			desc_act = np.round(action[sum(cts_action_space.shape):]).astype(int).tolist()
			next_state, reward, done, info = env.step([cts_act,desc_act])
			agent.perceive(state, action,reward,next_state, done, train=False)
			leng += 1

		print("Training {}".format(leng))
		for i in range(leng):
			if i % 20 == 0:
				print("\tIteration: {}".format(i))
			agent.train()


if __name__ == '__main__':
	main()
