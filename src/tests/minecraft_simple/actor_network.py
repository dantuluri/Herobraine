import tensorflow as tf
import numpy as np
import math

from common.utils import variable
from common.utils import make_lenet_embedding

# Hyper Parameters
REPLAY_BUFFER_SIZE = 1000000
LAYER1_SIZE = 10
LAYER2_SIZE = 10
LEARNING_RATE = 1e-4
TAU = 0.001
BATCH_SIZE = 64
GAMMA = 0.99

class ActorNetwork:
	"""docstring for ActorNetwork"""
	def __init__(self,sess,state_space,action_space,has_subcritics=False):
		#"""Constructor
		#param: subcritics : flag to turn on subcritic networks (critic networks per Q layer) """
		self.sess = sess
		self.state_space = state_space
		self.action_space = action_space
		self.has_subcritics = has_subcritics
		self.layers = []
		self.target_layers = []
		# create actor network
		self.state_input,self.action_output,self.net, convws = self.create_network(state_space,action_space)

		# create target actor network
		self.target_state_input,\
		self.target_action_output,\
		self.target_update,\
		self.target_net = self.create_target_network(state_space,action_space,self.net, convws)

		# define training rules
		self.create_training_method()

		self.sess.run(tf.initialize_all_variables())

		self.update_target()

		#self.load_network()

	def create_training_method(self):
		self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])
		self.parameters_gradients = tf.gradients(self.action_output,self.net,-self.q_gradient_input)
		self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(list(zip(self.parameters_gradients,self.net)))

	def create_network(self,state_space,action_space):
		layer1_size = LAYER1_SIZE
		layer2_size = LAYER2_SIZE
		
		state_dim = state_space.shape[0] 
		action_dim = self.action_dim = sum(action_space[0].shape)  + action_space[1].shape 

		embedding = state_input = tf.placeholder("float",[None] + list(state_space.shape))
		convws = []

		if len(state_space.shape) > 1:
			embedding, convws, state_dim = make_lenet_embedding(state_input)
		self.layers += [tf.identity(state_input)]

		W1 = variable([state_dim,layer1_size],state_dim)
		b1 = variable([layer1_size],state_dim)
		W2 = variable([layer1_size,layer2_size],layer1_size)
		b2 = variable([layer2_size],layer1_size)
		W3 = tf.Variable(tf.random_uniform([layer2_size,action_dim],-3e-3,3e-3))
		b3 = tf.Variable(tf.random_uniform([action_dim],-3e-3,3e-3))

		layer1 = tf.nn.relu(tf.matmul(embedding,W1) + b1)

		layer2 = tf.nn.relu(tf.matmul(layer1,W2) + b2)

		action_output = (tf.matmul(layer2,W3) + b3)

		return state_input,action_output,[W1,b1,W2,b2,W3,b3], convws


	def create_target_network(self,state_space,action_space,net, convws):
		
		ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
		target_update = ema.apply(net + convws)
		target_net = [ema.average(x) for x in net]
		conv_net = [ema.average(x) for x in convws]

		embedding = state_input = tf.placeholder("float",[None] + list(state_space.shape))

		if len(state_space.shape) > 1:
			embedding, convws, _ = make_lenet_embedding(state_input, conv_net)


		layer1 = tf.nn.relu(tf.matmul(embedding,target_net[0]) + target_net[1])

		layer2 = tf.nn.relu(tf.matmul(layer1,target_net[2]) + target_net[3])

		action_output = tf.tanh(tf.matmul(layer2,target_net[4]) + target_net[5])


		return state_input,action_output,target_update,target_net

	def update_target(self):
		self.sess.run(self.target_update)

	def train(self,q_gradient_batch,state_batch):

		self.sess.run(self.optimizer,feed_dict={
			self.q_gradient_input:q_gradient_batch,
			self.state_input:state_batch
			})

	def actions(self,state_batch):
		"""
		"""
		action_batch = self.sess.run(self.action_output,feed_dict={
			self.state_input:state_batch
			})
		return action_batch

	def action(self,state):
		""" Performs an action by propogating through the net"""
		action_output = self.sess.run(self.action_output,feed_dict={
			self.state_input:[state]
			})[0]
		return action_output


	def action_activations(self, state):
		""" Gets a pair of action, [state, layer1, layer2, ...] """
		output = self.sess.run([self.action_output,
			self.layers],feed_dict={
			self.state_input:[state]
			})
		action = output[0][0]
		activations = [activation[0] for activation in output[1]]
		return action, activations


	def target_actions(self,state_batch):
		""" Lag actor network """

		next_action_batch = self.sess.run(self.target_action_output,feed_dict={
			self.target_state_input:state_batch
			})
		return next_action_batch

	def target_action_activations(self, state_batch):
		next_action_batch = self.sess.run(
			[self.target_action_output,self.target_layers],feed_dict={
			self.target_state_input:state_batch
			})