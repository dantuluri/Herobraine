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
    PIXEL_RESOLUTION,
    LSTM_HIDDEN)

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
            self.actions = self.create_model()

        with tf.variable_scope("training"):
            self.label_ph, \
            self.loss, \
            self.train_op = self.create_training()

    
    # Current approach - use the tuple type to support arbitrary sequence lengths
    # Take the input (shape = [batch_size, sequence_len] + self.state_space) and flatten to
    # [sequence_len * batch_size] + self.state_space to do the convolutional layers and some
    # fully connected layers. Then map this back to [batch_size, sequence_len, num_neruons] 
    # and apply lstm layer using mapfn to introduce the remaining time independent layers 
    def create_model(self):
        """
        Creates the model.
        Returns: state_placeholder, action output tensor.
        """        
        
        # Input Placeholder (batch, time, [state])
        state_ph = tf.placeholder(tf.float32, shape=[None] + [None] + self.state_space)

        # Training Flag
        training_ph = tf.placeholder(tf.bool)

        # Hidden state Placeholder 
        batch_size    = tf.shape(state_ph)[0]
        #initial_state = tf.placeholder(tf.float32, shape=[None] + self.lstm_hidden_size)


        filter_size = max(self.state_space[:-1])

        head = state_ph

        # Reshape the data to flatten sequences
        head = tf.reshape(head, [-1, self.state_space])

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
        with tf.variable_scope("flatten_fc"):
            num_neurons = np.prod(head.get_shape().as_list()[1:])
            head = tf.reshape(head, [-1, num_neurons])

        # Apply some fc layers 
        for i, fc_size in enumerate(FC_SIZES):
            with tf.variable_scope("fc_{}".format(i)):
                head = tf.layers.dense(inputs=head, units=fc_size, activation=tf.nn.relu)

        # Reshape the data to retain sequence classification
        with tf.variable_scope("map_sequences"):
            num_neurons = np.prod(head.get_shape().as_list()[1:])
            head = tf.reshape(head, [batch_size, -1, num_neurons])

        # Introduce lstm layer
        with tf.variable_scope('lstm'):
            cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_HIDDEN, state_is_tuple=True)
            initial_state = cell.zero_state(batch_size, tf.float32)
            head, lstm_states = tf.nn.dynamic_rnn(cell, head, initial_state=initial_state, time_major=False)

        # Apply dropout
        dropout = lambda x : tf.layers.dropout(inputs=x, rate=DROPOUT, training=training_ph)
        head = tf.map_fn(dropout, head)

        with tf.variable_scope("fc_final"):
            # Calculate the dimensionality of action space
            num_outputs = sum(self.action_space)
            dense = lambda x : tf.layers.dense(inputs=x, units=num_outputs)
            head = tf.map_fn(dense, head)

        # Apply selective softmax accordin to various XOR conditions on the output
        with tf.variable_scope("action"):
            actions = []
            for _ in range(batch_size):
                action = []
                subspace_iter = 0

                for space in self.action_space:
                    logits = head[:, subspace_iter:subspace_iter+space]
                    # Note: we could also collect probabilities:
                    probabilities = tf.nn.softmax(logits)
                    argmax = tf.argmax(input=logits, axis=1)
                    
                    action.append((logits, argmax, probabilities))
                    subspace_iter += space

                actions.append(action)

        return state_ph, training_ph, actions

    def create_training(self):
        """
        Creates the training procedure.
        Returns: loss tensor, training_operation
        """    
        # Create the label placeholder
        label_ph = tf.placeholder(tf.int32, shape=[None, None, len(self.action_space)])
        sublabel = []
        with tf.variable_scope("label_processing"):
            # Convert it to a onehot.
            sublabels = []
            for b in range(batch_size):
                sublabel = []
                for i, space in enumerate(self.action_space):
                    with tf.variable_scope("one_hot_{}".format(i)):
                        sublabel.append(
                            tf.one_hot(indices=tf.cast(label_ph[b,:,i], tf.int32), depth=space))
                sublabels.append(sublabel)

        # First create the loss for each subspace.
        with tf.variable_scope("loss"):
            subloss = []
            for action, sublabel in zip(self.actions, sublabels)
                for ((logit_subspaces, _, _), labels) in zip(action, sublabel):
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
        # TODO support dynamic hidden state for inference
        cur_loss, _ = self.sess.run([self.loss, self.train_op], {
            self.training_ph: True,
            self.state_ph:  batch_states,
            self.label_ph:  batch_labels
            })
        return cur_loss

        # batch_states = [(h_i, S_i), (h_j, S_j), ...]
        # batch_labels = [L_i, L_j, ...]

        # cur_loss, _ = self.sess.run([self.loss, self.train_op], {
        #     self.training_ph: True,
        #     self.state_ph:  batch_states,
        #     self.hidden_ph: hidden_states,
        #     self.label_ph:  batch_labels
        #     })
        # return cur_loss

    def act(self, state):
        """
        Acts on single or multiple states.
        Returns actions in a onehot 
        """

        # TODO modify to properly handle sequence interactions

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

