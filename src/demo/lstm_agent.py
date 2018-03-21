


class Input(object):
    def __init__(self, batchsize, num_steps, data):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = batch_producer(data, batch_size, num_steps)

class Model(object):
    def __init__(self, input, is_training, hidden_size, vocab_size, num_layers, dropout = 0.5, init_scale = 0.05):
        self.is_training = is_training
        self.input_obj   = input
        self.batch_size  = input.batch_size
        self.num_steps   = input.num_steps 


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


    # Pass this batch_size manny sequences
    def train(sequence_list[5]):

        # Goal is to get (hidden_state, [chunk])

        #Get sequence from random shuffle

        # Take the each max_seq_len bundle ([batch_size, max_seq_len, input_size])
        for i in sequence_list:

            full_seq.split(500)

        # Call ses.run() on each chunk with the hidden state from previous chunk as activation




# create the word embeddings
with tf.device("/cpu:0"):
    embedding = tf.Variable(tf.random_uniform([vocab_size, self.hidden_size], -init_scale, init_scale))
    inputs = tf.nn.embedding_lookup(embedding, self.input_obj.input_data)

    if is_training and dropout < 1:
    inputs = tf.nn.dropout(inputs, dropout)