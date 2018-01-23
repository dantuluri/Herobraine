"""
Cloner.py -- The main program for cloning expert data from minecraft.
    1. Loads data from some bc_data store
    2. Starts a training loop and trains the model
    3. Starts a demonstration loop and runs a snapshot of  the model
       for a specified number of frames.
"""
import tensorflow as tf
import numpy as np
import os
import argparse
import threading

import gym
import minecraft_py
import gym_minecraft

from config import(
    GYM_RESOLUTION,
    MALMO_IP,
    BINDINGS)

# Import hyperparameters.
from config import (
    SMALLEST_FEATURE_MAP,
    FC_SIZES,
    LEARNING_RATE,
    DROPOUT)

def get_options():
    parser = argparse.ArgumentParser(description='Clone some expert data..')
    parser.add_argument('bc_data', type=str,
        help="The main datastore for this particular expert.")

    args = parser.parse_args()
    pass

class Agent:
    """
    The main agent class. This instantiates a tensorflow model
    and provides training/demosntration functionality, given some action
    space.
    """

    # TODO: MOVE AGENT TO ITS OWN FILE.
    def __init__(self, env, state_space, action_space, sess):
        self.sess = sess
        self.env = env
        self.action_space = action_space
        self.state_space = state_space

        self.state_ph, \
        self.training_ph, \
        self.action = self.create_model()

        self.label_ph, \
        self.loss, \
        self.train_op = self.create_training()
    

    def create_model(self):
        """
        Creates the model.
        Returns: state_placeholder, action output tensor.
        """
        state_ph = tf.placeholder(tf.float32, shape=[None] + self.state_space)
        training_ph = tf.placeholder(tf.bool)

        filter_size = max(self.state_space[:-1])
        head = state_ph

        # Apply some convolutions.
        conv_iter = 0
        while filter_size > SMALLEST_FEATURE_MAP:
            # Build a convolution
            with tf.variable_scope("conv_block_{}".format(conv_iter)):
                head = tf.layers.conv2d(
                    inputs=head,
                    filters=min(2**conv_iter*16, 64),
                    kernel_size=[5, 5],
                    padding="same",
                    activation=tf.nn.relu)
                filter_size = min(head.get_shape().as_list()[1:-1])

                if filter_size > SMALLEST_FEATURE_MAP:
                    # Apply pooling
                    head = tf.layers.max_pooling2d(inputs=head, pool_size=[2, 2], strides=2)

            conv_iter += 1
            filter_size = min(head.get_shape().as_list()[1:-1])
            
        # Flatten to fully connected
        with tf.variable_scope("flatten"):
            num_neurons = np.prod(head.get_shape().as_list()[1:])
            head = tf.reshape(head, [-1, num_neurons])

        # Apply some fc layers
        for i, fc_size in enumerate(FC_SIZES):
            with tf.variable_scope("fc_{}".format(i)):
                head = tf.layers.dense(inputs=head, units=fc_size, activation=tf.nn.relu)

        # Apply dropout
        head = tf.layers.dropout(
            inputs=head, rate=DROPOUT, training=training_ph)

        with tf.variable_scope("fc_final"):
            # Calculate the dimensionality of action space
            num_outputs = sum(self.action_space)
            head = tf.layers.dense(inputs=head, units=num_outputs)

        # Apply selective softmax accordin to various XOR conditions on the output
        with tf.variable_scope("action"):
            action = []
            subspace_iter = 0

            for space in self.action_space:
                logits = head[:, subspace_iter:subspace_iter+space]
                # Note: we could also collect probabilities:
                # probabilities = tf.nn.softmax(logits)
                argmax = tf.nn.tf.argmax(input=logits, axis=1)
                
                action.append((logits, argmax))
                subspace_iter += space

        return state_ph, training_ph, action

    def create_training(self):
        """
        Creates the training procedure.
        Returns: loss tensor, training_operation
        """
        with tf.variable_scope("training"):
            # Create the label placeholder
            label_ph = tf.placeholder(tf.int32, shape=[None, len(self.action_space)])
            sublabel = []
            with tf.variable_scope("label_processing"):
                # Convert it to a onehot.
                for i, space in enumerate(self.action_space):
                    with tf.variable_scope("one_hot_{}".format(i)):
                        sublabel.append(
                            tf.one_hot(indices=tf.cast(label_ph[:,i], tf.int32), depth=space))

            # First create the loss for each subspace.
            with tf.variable_scope("loss"):
                subloss = []
                for ((logit_subspace, _), label) in zip(self.action, sublabel):
                    with tf.variable_scope("subloss_{}".format(len(subloss))):
                        subloss.append(
                            tf.losses.softmax_cross_entropy(
                                onehot_labels=label, logits=logit_subspace))

                # Integrate the loss
                loss = tf.add_n(subloss, name="loss")

            with tf.variable_scope("optimization"):
                optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
                train_op = optimizer.minimize(loss=loss)

        return label_ph, loss, train_op

    def train(self, batch_states, batch_labels):
        """
        Trains the model
        """
        cur_loss, _ = self.sess.run([self.loss, self.train_op], {
            self.training_ph: True,
            self.state_ph: batch_states,
            self.label_ph: batch_labels
            })
        return cur_loss

    def act(self, state):
        """
        Acts on single or multiple states.
        Returns actions in a onehot 
        """
        # Handle shape size update if single state
        assert len(state.shape) >= 3
        
        single_state = len(state.shape) == 3
        if single_state:
            state = np.expand_dims(state, axis=0)
        

        # Feed the state and get the action onehot
        _, argmax_subspace = zip(*self.action)
        subspace_action_argmax = self.sess.run(argmax_subspace, {
            self.state_ph: state,
            self.training_ph: False
            })

        # If it's a single state, return a single action
        # not a list of actions.
        if single_state:
            subspace_action_argmax = [
                subspace[0] for subspace in subspace_action_argmax]

        return subspace_action_argmax





def run_training(coord, agent, bc_data_dir):
    """
    Runs training for the agent.
    """
    # Load the file store.
    # Train the agent from the file store
    #   # Persist the agent occasionally in the same directory.
    pass


def run_demonstrations(coord, env, agent):
    """
    Runs the main demonstration for the agent.
    """


    pass


def run_main(opts):
    # Create the environment with specified arguments
    env = gym.make('MinecraftDefaultWorld1-v0')
    env.init(
        start_minecraft=None,
        client_pool=[MALMO_IP],
        continuous_discrete = True,
        videoResolution=GYM_RESOLUTION,
        add_noop_command=True)

    # Define the action space. (length of key bindings plus null action)
    action_space = [len(d) + 1for d,_ in BINDINGS]
    # Define the state space.
    state_space = list(GYM_RESOLUTION) + [3] # 3 color channels

    # Create a model
    sess = tf.InteractiveSession()
    agent = Agent(env, state_space, action_space, sess)
    sess.run(tf.global_variables_initializer())

    # Start the training thread with a coordinator.
    coord = tf.train.Coordinator()
    training_thread = threading.Thread(
        target=run_training, args=(coord, agent, opts.bc_data))
    training_thread.start()

    # Begin performing demonstrations.
    run_demonstrations(coord, env, agent)
    coord.join([threading_thread])

if __name__ == "__main__":
    # Parse arguments
    opts = get_options()
    # Start the main thread.
    run_main(opts)