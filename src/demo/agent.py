"""
agent.py -- The main module for describing the behacioral cloning agent.
"""
import tensorflow as tf
import numpy as np
import datetime
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
            self.initial_state, \
            self.hidden_state, \
            self.actions = self.create_model()

        with tf.variable_scope("target_model"):
            self.target_state_ph, \
            self.target_training_ph, \
            self.target_initial_state, \
            self.target_hidden_state, \
            self.target_actions = self.create_model(summaries=False)

        with tf.variable_scope("training"):
            self.label_ph, \
            self.loss, \
            self.train_op = self.create_training()


        with tf.variable_scope("copy_operation"):
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')
            target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_model')

            # Assuming that the vars are sorted correctly
            self.target_update = [
                tv.assign(v.value()) for tv, v in zip(target_vars, vars)
                ]

        self.merge = tf.summary.merge_all()
        

    def get_state_variables(self, batch_size, cell):
        # For each layer, get the initial state and make a variable out of it
        # to enable updating its value.
        state_variables = \
            tf.contrib.rnn.LSTMStateTuple(
                tf.Variable(state_c, trainable=False),
                tf.Variable(state_h, trainable=False))
        # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
        return tuple(state_variables)


    def get_state_update_op(self, state_variables, new_states):
        # Add an operation to update the train states with the last state tensors
        update_ops = [state_variable[0].assign(new_state[0]),
                            state_variable[1].assign(new_state[1])]

        return update_ops

    
    # Current approach - use the tuple type to support arbitrary sequence lengths
    # Take the input (shape = [batch_size, sequence_len] + self.state_space) and flatten to
    # [sequence_len * batch_size] + self.state_space to do the convolutional layers and some
    # fully connected layers. Then map this back to [batch_size, sequence_len, num_neruons] 
    # and apply lstm layer using mapfn to introduce the remaining time independent layers 
    def create_model(self, summaries=True):
        """self.
        Creates the model.
        Returns: state_placeholder, action output tensor.
        """        
        
        # Input Placeholder (batch, time, [state])
        state_ph = tf.placeholder(tf.float32, shape=[None] + [None] + self.state_space, name="state")

        # Training Flag
        training_ph = tf.placeholder(tf.bool)

        # Hidden state Placeholder 
        batch_size    = tf.shape(state_ph)[0]
        # initial_state_ph = tf.placeholder(tf.float32, shape=[2, None, LSTM_HIDDEN], name="initial_state_ph")
        # initial_state = tf.contrib.rnn.LSTMStateTuple(initial_state_ph[0], initial_state_ph[1])


        filter_size = max(self.state_space[:-1])

        head = state_ph
        print(head.get_shape())
        print("Downsampling.")
        head = tf.layers.max_pooling3d(inputs=head, pool_size=[1,4, 4], strides=[1,4,4])
        print("Done", head.get_shape())

        # # Reshape the data to flatten sequences    
        #shape = tf.shape(head) 
        #head = tf.reshape(head, [np.prod(shape[:2]), )

        # Normalize the data
        with tf.variable_scope("normalization"):
            head /= PIXEL_RESOLUTION
            head -= 0.5 # TODO Take the mean of the data.

        # Apply some convolutions.
        conv_iter = 0
        while filter_size > SMALLEST_FEATURE_MAP:
            # Build a convolution
            with tf.variable_scope("conv_block_{}".format(conv_iter)):
                print("Layer i")
                print(head.get_shape())
                conv_fn = lambda x : \
                    tf.layers.conv3d(
                        inputs=x,
                        filters=min(2**conv_iter*32, 64),
                        kernel_size=[1, 5, 5],
                        padding="same",
                        activation=tf.nn.relu)
                head = conv_fn(head)
                filter_size = min(head.get_shape().as_list()[2:-1])
                if summaries: tf.summary.histogram("activations_{}".format(conv_iter), head)

                if filter_size > SMALLEST_FEATURE_MAP:
                    # Apply pooling
                    pooling_fn = lambda x : tf.layers.max_pooling3d(inputs=x, pool_size=[1,2, 2], strides=[1,2,2])
                    head = pooling_fn(head)

            conv_iter += 1
            filter_size = min(head.get_shape().as_list()[2:-1])
            
        # Flatten to fully connected 
        print(head.get_shape())
        with tf.variable_scope("flatten_fc"):
            shape = tf.shape(head)
            num_neurons = np.prod(head.get_shape().as_list()[2  :])
            flatten_fn = lambda x : tf.reshape(x, [shape[0], shape[1], num_neurons])
            head = flatten_fn(head)
            if summaries: tf.summary.histogram("activations_flatten", head)


        # Apply some fc layers 
        print(head.get_shape())
        for i, fc_size in enumerate(FC_SIZES[:2]):
            with tf.variable_scope("fc_{}".format(i)):
                #fc_fn = lambda x :  tf.layers.dense(inputs=x, units=fc_size, activation=tf.nn.relu)
                #head = tf.map_fn(fc_fn, head)
                head = tf.layers.dense(inputs=head, units=fc_size, activation=tf.nn.relu)
        
        # Introduce lstm layer
        with tf.variable_scope('lstm'):
            cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_HIDDEN, state_is_tuple=True)
            initial_state = cell.zero_state(batch_size, tf.float32)

            #print (cell.state_size)
            #print(head.get_shape())
            head, hidden_state = tf.nn.dynamic_rnn(
                cell, 
                head, 
                initial_state=initial_state, 
                time_major=False, 
                swap_memory=True)

        print(head.get_shape())


        print(head.get_shape())
        # Apply dropout
        head = tf.layers.dropout(inputs=head, rate=DROPOUT, training=training_ph)
        

        with tf.variable_scope("fc_final"):
            # Calculate the dimensionality of action space
            num_outputs = sum(self.action_space)
            head = tf.layers.dense(inputs=head, units=num_outputs)
            #print(head.get_shape())


        # Apply selective softmax accordin to various XOR conditions on the output
        with tf.variable_scope("action"):
            #actions = tf.map_fn(self.selective_softmax, head, dtype = tf.float32)
            action = [] 
            subspace_iter = 0

            for space in self.action_space: #, is_discrete
                logits = head[:,:, subspace_iter:subspace_iter+space]
                print(tf.shape(logits))

                # if is_discrete:
                    # Note: we could also collect probabilities:
                probabilities = tf.nn.softmax(logits, axis=-1)
                argmax = tf.argmax(input=logits, axis=-1)
                # else:
                    # probabilities, argmax = None, None
                action.append((logits, argmax, probabilities))
                subspace_iter += space

        return state_ph, training_ph, initial_state,  hidden_state, action #TODO ,update_hidden_op

    def create_training(self):
        """
        Creates the training procedure.
        Returns: loss tensor, training_operation
        """    

        label_ph = tf.placeholder(tf.int32, shape=[None, None, len(self.action_space)])
        sublabel = []
        with tf.variable_scope("label_processing"):
            # Convert it to a onehot.
            sublabel = []
            for i, space in enumerate(self.action_space):
                with tf.variable_scope("one_hot_{}".format(i)):
                    sublabel.append(tf.one_hot(indices=tf.cast(label_ph[:,:,i], tf.int32), depth=space)) 

        # First create the loss for each subspace.
        with tf.variable_scope("loss"):
            subloss = []
            for ((logit_subspace, _, _), label) in zip(self.actions, sublabel):
                with tf.variable_scope("subloss_{}".format(len(subloss))):
                    subloss.append(
                        tf.nn.softmax_cross_entropy_with_logits(
                            labels=label, logits=logit_subspace, dim=-1))

            # Integrate the loss
            loss = tf.add_n(subloss)/float(len(subloss))
            loss = tf.reduce_mean(subloss, name="loss")
            tf.summary.scalar("Loss", loss)
            print(loss)

            # Adjust for sequence length.

        with tf.variable_scope("optimization"):
            optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
            train_op = optimizer.minimize(loss=loss)

        return label_ph, loss, train_op

    def train(self, batch_states, batch_labels, train_writer, tick):
        """
        Trains the model
        """
        # TODO support dynamic hidden state for inference


        cur_loss, _, summary = self.sess.run([self.loss, self.train_op, self.merge], {
            self.training_ph: True,
            self.state_ph:  batch_states,
            self.label_ph:  batch_labels
            })


        train_writer.add_summary(summary, tick)
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

    def act(self, state, hidden, update=False):
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
            state = np.expand_dims(np.expand_dims(state, axis=0), axis=0)
        
        # If update
        fd = {
            self.target_state_ph: state,
            self.target_training_ph: False,
        }

        if update:
            self.sess.run(self.target_update)
        else:
            fd.update({self.target_initial_state: hidden})
   

        # Feed the state and get the action onehot
        _, _, probability_subspace = zip(*self.target_actions)
        subspace_action_prob, out_hidden = self.sess.run([probability_subspace, self.target_hidden_state], fd)

        subspace_action_argmax = [[
            [np.argmax(np.random.multinomial(1, pvals=v - max(sum(v) -1,0))) for v in sap] 
            for sap in ssap] for ssap in subspace_action_prob]

        # If it's a single state, return a single action
        # not a list of actions.
        if single_state:
            subspace_action_argmax = [
                subspace[0] for subspace in subspace_action_argmax]


        
        return subspace_action_argmax, out_hidden

