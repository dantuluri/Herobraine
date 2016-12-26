import tensorflow as tf
import math
import operator

from functools import reduce 

TRACK_VARS=False

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    # with tf.name_scope('summaries'):
    #     mean = tf.reduce_mean(var)
    #     tf.scalar_summary('mean/' + name, mean)
    #     with tf.name_scope('stddev'):
    #         stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    #     tf.scalar_summary('stddev/' + name, stddev)
    #     tf.scalar_summary('max/' + name, tf.reduce_max(var))
    #     tf.scalar_summary('min/' + name, tf.reduce_min(var))
    #     tf.histogram_summary(name, var)
    pass

def variable(shape,f, name="Variable"):
    """
    Creates a tensor of SHAPE drawn from
    a random uniform distribution in 0 +/- 1/sqrt(f)
    """
    #TODO: fix this. currently shape is a [Dimension, int] object
    v =  tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)), name=name)
    #v = tf.Variable(tf.constant(-0.00001, shape=shape, name=name))
    if TRACK_VARS: variable_summaries(var, name)
    return v

def make_lenet_embedding(input_tensor, weights=[]):
    """
    Defines a lenet embedding for minecraft.
    """

    with tf.variable_scope("lenet_embedding"):
        if not weights:
            weights.append(variable([5,5, 3, 16], 16, "conv_kernel_1"))
            weights.append(variable([16], 16, "bias_1"))
            weights.append(variable([5,5, 16, 32], 32, "conv_kernel_2"))
            weights.append(variable([32], 32, "bias_2"))
            weights.append(variable([5,5, 32, 32], 32, "conv_kernel_3"))
            weights.append(variable([32], 32, "bias_3"))
            # weights.append(variable([5,5, 32, 16], 16, "conv_kernel_4"))
            # weights.append(variable([16], 16, "bias_4"))

        with tf.variable_scope("layer1"):
            conv1 =  tf.nn.conv2d(input_tensor, weights[0], [1,1,1,1], padding='SAME')
            layer1 = tf.nn.relu(conv1+weights[1])


        with tf.variable_scope("layer2"):
            pool1 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pool1')
            layer2 = tf.nn.lrn(pool1, 4, bias=1.0,   alpha=0.001 / 9.0, beta=0.75,
                              name='layer2')


        with tf.variable_scope("layer3"):
            conv2 = tf.nn.conv2d(layer2, weights[2], [1,1,1,1], padding='SAME')
            layer3 = tf.nn.relu(conv2+weights[3])


        with tf.variable_scope("layer4"):
            pool2 = tf.nn.max_pool(layer3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pool1')
            layer4 = tf.nn.lrn(pool2, 4, bias=1.0,   alpha=0.001 / 9.0, beta=0.75,
                                  name='layer4')

        with tf.variable_scope("layer5"):
            conv3 = tf.nn.conv2d(layer4, weights[4], [1,1,1,1], padding='SAME')
            layer5 = tf.nn.relu(conv3+weights[5])


        # with tf.variable_scope("layer6"):
        #     pool3 = tf.nn.max_pool(layer5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        #                            padding='SAME', name='pool1')
        #     layer6 = tf.nn.lrn(pool3, 4, bias=1.0,   alpha=0.001 / 9.0, beta=0.75,
        #                   name='layer6')

        # with tf.variable_scope("layer7"):
        #     conv4 = tf.nn.conv2d(layer6, weights[6], [1,1,1,1], padding='SAME')
        #     layer7 = tf.nn.relu(conv4+weights[7])


        shape = layer5.get_shape().as_list()
        out_dim = reduce(operator.mul, shape[1:], 1)
        # Could change to theglobal average pooling operation.
        reshaped_output = tf.reshape(layer5, [-1, out_dim])

    return reshaped_output, weights, out_dim