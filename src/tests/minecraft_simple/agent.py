###
# A simple DDPG agent for minecraft.
###
# -----------------------------------
# Deep Deterministic Policy Gradient
# Author: Flood Sung
# Date: 2016.5.4
# -----------------------------------
import gym
import tensorflow as tf
import numpy as np
from common.ou_noise import OUNoise
from critic_network import CriticNetwork
from actor_network import ActorNetwork
from common.replay_buffer import ReplayBuffer
from common.mdr import MultiDiscreteRandom


# Hyper Parameters:

REPLAY_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 64
EXPLORATION_INITIAL= 1e-6
GAMMA = 0.9


class DDPG:
    """docstring for DDPG"""
    def __init__(self, sess, state_space, action_space):
        self.name = 'DDPG' 
        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_space = state_space
        self.action_space = action_space
        self.sess = sess

        with tf.variable_scope("Actor"):
            self.actor_network = ActorNetwork(self.sess, self.state_space, self.action_space)
        with tf.variable_scope("Critic"):
            self.critic_network = CriticNetwork(self.sess, self.state_space, self.action_space)

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, uniform=True)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        # Taylored directly to minecraft.
        self.cts_action_dim = sum(self.action_space[0].shape)
        self.desc_action_dim = self.action_space[1].shape
        self.action_dim = self.cts_action_dim + self.desc_action_dim
        self.cts_exploration_noise = OUNoise(self.cts_action_dim)
        self.desc_exploration_noise = MultiDiscreteRandom(self.desc_action_dim, EXPLORATION_INITIAL)

    def train(self):
        if self.replay_buffer.count() >  REPLAY_START_SIZE:
            # Sample a random minibatch of N transitions from replay buffer
            minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
            state_batch = np.asarray([data[0] for data in minibatch])
            action_batch = np.asarray([data[1] for data in minibatch])
            reward_batch = np.asarray([data[2] for data in minibatch])
            next_state_batch = np.asarray([data[3] for data in minibatch])
            done_batch = np.asarray([data[4] for data in minibatch])

            ######################
            # Do NORMAL Q LEARNING
            ######################
            # for action_dim = 1
            action_batch = np.resize(action_batch,[BATCH_SIZE,self.action_dim])

            # Calculate y_batch
            next_action_batch = self.actor_network.target_actions(next_state_batch)
            q_value_batch = self.critic_network.target_q(next_state_batch, next_action_batch)

            y_batch = []
            for i in range(len(minibatch)):
                if done_batch[i]:
                    y_batch.append(reward_batch[i])
                else :n
                    y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
            y_batch = np.resize(y_batch,[BATCH_SIZE,1])

            # Update critic by minimizing the loss L
            self.critic_network.train(y_batch,state_batch,action_batch)
            ######################

            # Update the actor policy using the sampled gradient:
            action_batch_for_gradients = self.actor_network.actions(state_batch)
            q_gradient_batch = self.critic_network.gradients(state_batch, action_batch_for_gradients)

            self.actor_network.train(q_gradient_batch,state_batch)

            # Update the target networks
            self.actor_network.update_target()
            self.critic_network.update_target()

    def action(self,state):
        action = self.actor_network.action(state)
        return action

    def noise_action(self,  state):
        action = self.actor_network.action(state)
        cts_action = action[:self.cts_action_dim] + env.cts_exploration_noise.noise()
        desc_action = self.desc_exploration_noise.noise(action[self.cts_action_dim:])
        action = np.append(cts_action, desc_action)
        return action

    def perceive(self,state, action,reward,next_state, done, train=True):
        # Store transition (s_t,v_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(
            state,action,reward,next_state, done)

        # Store transitions to replay start size then start training
        if  train:
            self.train()

        # Re-iniitialize the random process when an episode ends
        if done:
            self.cts_exploration_noise.reset()
            self.desc_exploration_noise.reset()