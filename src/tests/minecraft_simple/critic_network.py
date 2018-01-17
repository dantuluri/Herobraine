
import tensorflow as tf
import numpy as np
import math

from common.utils import variable
from common.utils import variable_summaries
from common.utils import make_lenet_embedding

LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 1e-3
TAU = 0.001
L2 = 0.01

class CriticNetwork:
	"""docstring for CriticNetwork"""
	def __init__(self,sess,state_space,action_space):
		self.time_step = 0
		self.sess = sess
		self.state_space = state_space
		self.action_space = action_space
		# create q network
		self.state_input,\
		self.action_input,\
		self.q_value_output,\
		self.net, \
		self.convws = self.create_q_network(state_space,action_space)

		# create target q network (the same structure with q network)
		self.target_state_input,\
		self.target_action_input,\
		self.target_q_value_output,\
		self.target_update = self.create_target_q_network(state_space,action_space,self.net, self.convws)

		self.create_training_method()

		# initialization
		self.sess.run(tf.initialize_all_variables())

		self.update_target()


	def create_training_method(self):
		# Define training optimizer
		self.y_input = tf.placeholder("float",[None,1])
		weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
		self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
		self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
		self.action_gradients = tf.gradients(self.q_value_output,self.action_input)

	def create_q_network(self,state_space,action_space):

		# the layer size could be changed
		layer1_size = LAYER1_SIZE
		layer2_size = LAYER2_SIZE
		# Cointinuous + discrete
		action_dim = sum(action_space[0].shape)  + action_space[1].shape 
		 #If the input is two dimensional, then this will get overwritten
		state_dim = state_space.shape[0] 

		embedding = state_input = tf.placeholder("float",[None] + list(state_space.shape))
		action_input = tf.placeholder("float",[None,action_dim])
		convws = []

		if len(state_space.shape) > 1:
			embedding, convws, state_dim = make_lenet_embedding(state_input)

		W1 = variable([state_dim,layer1_size],state_dim)
		b1 = variable([layer1_size],state_dim)
		W2 = variable([layer1_size,layer2_size],layer1_size+action_dim)
		W2_action = variable([action_dim,layer2_size],layer1_size+action_dim)
		b2 = variable([layer2_size],layer1_size+action_dim)
		W3 = tf.Variable(tf.random_uniform([layer2_size,1],-3e-3,3e-3))
		b3 = tf.Variable(tf.random_uniform([1],-3e-3,3e-3))

		layer1 = tf.nn.relu(tf.matmul(embedding,W1) + b1)
		layer2 = tf.nn.relu(tf.matmul(layer1,W2) + tf.matmul(action_input,W2_action) + b2)
		q_value_output = tf.identity(tf.matmul(layer2,W3) + b3)
		variable_summaries(q_value_output, "Critic_Q")
		return state_input,action_input, \
			   q_value_output,[W1,b1,W2,W2_action,b2,W3,b3], convws

	def create_target_q_network(self,state_space,action_space,net, convws):

		ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
		target_update = ema.apply(net + convws)
		target_net = [ema.average(x) for x in net]
		conv_net = [ema.average(x) for x in convws]

		embedding = state_input = tf.placeholder("float",[None] + list(state_space.shape))

		action_dim = sum(action_space[0].shape)  + action_space[1].shape 
		action_input = tf.placeholder("float",[None,action_dim])

		if len(state_space.shape) > 1:
			embedding, convws, _ = make_lenet_embedding(state_input, conv_net)


		layer1 = tf.nn.relu(tf.matmul(embedding,target_net[0]) + target_net[1])
		layer2 = tf.nn.relu(tf.matmul(layer1,target_net[2]) + tf.matmul(action_input,target_net[3]) + target_net[4])
		q_value_output = tf.identity(tf.matmul(layer2,target_net[5]) + target_net[6])
		variable_summaries(q_value_output, "Critic_T_Q")
		return state_input,action_input,q_value_output,target_update

	def update_target(self):
		self.sess.run(self.target_update)

	def train(self,y_batch,state_batch,action_batch):
		self.time_step += 1
		self.sess.run(self.optimizer,feed_dict={
			self.y_input:y_batch,
			self.state_input:state_batch,
			self.action_input:action_batch
			})

	def gradients(self,state_batch,action_batch):
		return self.sess.run(self.action_gradients,feed_dict={
			self.state_input:state_batch,
			self.action_input:action_batch
			})[0]

	def target_q(self,state_batch,action_batch):
		return self.sess.run(self.target_q_value_output,feed_dict={
			self.target_state_input:state_batch,
			self.target_action_input:action_batch
			})

	def q_value(self,state_batch,action_batch):
		return self.sess.run(self.q_value_output,feed_dict={
			self.state_input:state_batch,
			self.action_input:action_batch})