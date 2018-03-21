


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

# create the word embeddings
with tf.device("/cpu:0"):
    embedding = tf.Variable(tf.random_uniform([vocab_size, self.hidden_size], -init_scale, init_scale))
    inputs = tf.nn.embedding_lookup(embedding, self.input_obj.input_data)

    if is_training and dropout < 1:
    inputs = tf.nn.dropout(inputs, dropout)