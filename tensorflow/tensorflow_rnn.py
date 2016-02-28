# modified version from
# https://www.reddit.com/r/MachineLearning/comments/3sok8k/tensorflow_basic_rnn_example_with_variable_length/
import tensorflow as tf
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn.rnn_cell import BasicLSTMCell, LSTMCell
import numpy as np
import time

if __name__ == '__main__':
    np.random.seed(1)
    batch_size= 128
    n_steps = 64
    #seq_width = 128
    #seq_width = 256
    #seq_width = 512
    #seq_width = 1024
    seq_width = 2048
    n_vocab = 10000
    n_class = 5

    initializer = tf.random_uniform_initializer(-1,1)

    seq_input = tf.placeholder(tf.int32, [n_steps, batch_size])
    y = tf.placeholder(tf.int64, [batch_size, ])
    #sequence we will provide at runtime
    early_stop = tf.placeholder(tf.int32)
    #what timestep we want to stop at
    embedding_matrix = tf.Variable(tf.zeros([n_vocab, seq_width]))
    W_class = tf.Variable(tf.zeros([seq_width, n_class]))
    B_class = tf.Variable(tf.zeros([n_class, ]))

    embed_input = tf.nn.embedding_lookup(embedding_matrix, seq_input)

    inputs = [tf.reshape(i, (batch_size, seq_width)) for i in tf.split(0, n_steps, embed_input)]
    #inputs for rnn needs to be a list, each item being a timestep.
    #we need to split our input into each timestep, and reshape it because split keeps dims by default

    #cell = LSTMCell(seq_width, seq_width, initializer=initializer)
    cell = LSTMCell(seq_width, seq_width, initializer=initializer)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, states = rnn.rnn(cell, inputs, initial_state=initial_state, sequence_length=early_stop)

    last = outputs[-1]
    logit = tf.matmul(last, W_class) + B_class
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logit, y))

    iop = tf.initialize_all_variables()
    #create initialize op, this needs to be run by the session!
    #session = tf.Session()
    session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    session.run(iop)
      #actually initialize, if you don't do this you get errors about uninitialized stuff

    feed = {early_stop:n_steps,
            seq_input:np.random.randint(0, n_vocab, (n_steps, batch_size)).astype('int32'),
            y:np.random.randint(0, n_class, (batch_size)).astype("int64")}
      #define our feeds.
      #early_stop can be varied, but seq_input needs to match the shape that was defined earlier

    opt = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    #warmup
    for x in range(2):
        outs = session.run(opt, feed_dict=feed)

    t_start = time.time()
    for i in range(50):
      outs = session.run(opt, feed_dict=feed)
    tdiff = time.time() - t_start
    print "50 steps took:", tdiff
    print "per step:", tdiff / 50.
