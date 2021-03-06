"""
agent.py -- The main module for describing the behacioral cloning agent.
"""
import tensorflow as tf
import numpy as np
# Import hyperparameters.
from config import (
    SMALLEST_FEATURE_MAP,
    FC_SIZES,
    LEARNING_RATE,
    DROPOUT,
    PIXEL_RESOLUTION)

class Agent:
    """
    The main agent class. This instantiates a tensorflow model
    and provides training/demosntration functionality, given some action
    space.
    """

    # TODO: MOVE AGENT TO ITS OWN FILE.
    def __init__(self, state_space, action_space, sess):
        self.sess = sess
        self.action_space = action_space
        self.state_space = state_space

        with tf.variable_scope("model"):
            self.state_ph, \
            self.training_ph, \
            self.action = self.create_model()

        with tf.variable_scope("training"):
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

        # Normalize the data
        with tf.variable_scope("normalization"):
            head /= PIXEL_RESOLUTION
            head -= 0.5 # TODO Take the mean of the data.

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
                probabilities = tf.nn.softmax(logits)
                argmax = tf.argmax(input=logits, axis=1)
                
                action.append((logits, argmax, probabilities))
                subspace_iter += space

        return state_ph, training_ph, action

    def create_training(self):
        """
        Creates the training procedure.
        Returns: loss tensor, training_operation
        """    
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
            for ((logit_subspace, _, _), label) in zip(self.action, sublabel):
                with tf.variable_scope("subloss_{}".format(len(subloss))):
                    subloss.append(
                        tf.losses.softmax_cross_entropy(
                            onehot_labels=label, logits=logit_subspace))

            # Integrate the loss
            loss = tf.add_n(subloss, name="loss")/float(len(subloss))

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
        # If observations missed.
        assert len(state.shape) >= 3

        # Handle shape size update if single state
        single_state = len(state.shape) == 3
        if single_state:
            state = np.expand_dims(state, axis=0)
        

        # Feed the state and get the action onehot
        _, _, probability_subspace = zip(*self.action)
        subspace_action_prob = self.sess.run(probability_subspace, {
            self.state_ph: state,
            self.training_ph: False
            })

        subspace_action_argmax = [
            [np.argmax(np.random.multinomial(1, pvals=v - max(sum(v) -1,0))) for v in sap] 
            for sap in subspace_action_prob]

        # If it's a single state, return a single action
        # not a list of actions.
        if single_state:
            subspace_action_argmax = [
                subspace[0] for subspace in subspace_action_argmax]

        return subspace_action_argmax

