# modified version from
# http://deeplearning.net/tutorial/lstm.html

import six.moves.cPickle as pickle

from collections import OrderedDict
import sys
import time

import numpy as np
import theano
from theano import config
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

#theano.config.compute_test_value = 'warn'

# Set the random number generators' seeds for consistency
SEED = 123
np.random.seed(SEED)

def floatX(data):
    return np.asarray(data, dtype=config.floatX)

def param(shape, name):
    return theano.shared(floatX(np.random.randn(*shape)*0.2), name=name)

#proj_dim = 128
#proj_dim = 256
#proj_dim = 512
#proj_dim = 1024
proj_dim = 2048
embed_dim = proj_dim
batch_size = 128

n_class = 5
n_vocab = 10000

W_class = param((proj_dim, n_class), "W_class")
B_class = param((n_class, ), "B_class")

W_embed = param((n_vocab, embed_dim), "W_embed")
W_hh = param((proj_dim, proj_dim*4), "W_hh")

# for 4 lstm gates
W_toinp = param((proj_dim, proj_dim*4), "W_toinp")
B_toinp = param((proj_dim*4, ), "B_toinp")

def lstm_layer(inp, mask):

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = T.dot(h_, W_hh) + x_

        i = T.nnet.sigmoid(_slice(preact, 0, proj_dim))
        f = T.nnet.sigmoid(_slice(preact, 1, proj_dim))
        o = T.nnet.sigmoid(_slice(preact, 2, proj_dim))
        c = T.tanh(_slice(preact, 3, proj_dim))

        c = f * c_ + i * c

        h = o * T.tanh(c)

        # masking
        c = m_[:, None] * c + (1. - m_)[:, None] * c_
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    rval, updates = theano.scan(_step,
                                sequences=[mask, inp],
                                outputs_info=[T.alloc(floatX(0.),
                                                           inp.shape[1],
                                                           proj_dim),
                                              T.alloc(floatX(0.),
                                                           inp.shape[1],
                                                           proj_dim)],
                                name="lstm_scan")
    return rval[0]

def build_model():
    trng = RandomStreams(SEED)

    x = T.matrix('x', dtype='int64')
    x.tag.test_value = np.random.randint(0, n_vocab, size=(64, 128))

    mask = T.matrix('mask', dtype=config.floatX)
    mask.tag.test_value = np.ones((64, 128), dtype="float32")
    y = T.vector('y', dtype='int64')
    y.tag.test_value = np.random.randint(0, n_class, size=(128,))

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    embed = W_embed[x.flatten()].reshape([n_timesteps,
                                        n_samples,
                                        embed_dim])
    toinp = (T.dot(embed, W_toinp) + B_toinp)

    output = lstm_layer(toinp, mask)

    last = output[0]

    pred = T.nnet.softmax(T.dot(last, W_class) + B_class)

    cost = T.nnet.categorical_crossentropy(pred, y).mean()
    params = [W_class, B_class, W_embed, W_hh]

    updates = []
    for p, g in zip(params, theano.gradient.grad(cost, params)):
        updates.append((p, p-g*0.001))

    return x, y, mask, cost, pred, updates


def benchmark_lstm():
    print "starting to build model"
    t_start = time.time()
    x, y, mask, cost, pred, updates = build_model()
    f_update_shared = theano.function([x, mask, y], cost, updates=updates,
                                    name='full')
    print "done, took:", time.time() - t_start
    t_start = time.time()

    x_val = np.random.randint(0, n_vocab, size=(64, batch_size))
    y_val = np.random.randint(0, n_class, size=(batch_size ,))
    mask_val = np.ones((64, batch_size), dtype="float32")
    for i in range(50):
        print i
        cost_val = f_update_shared(x_val, mask_val, y_val)
    tdiff = time.time() - t_start
    print "done with 50 loop, took:", tdiff
    print tdiff / 50.

if __name__ == '__main__':
    benchmark_lstm()

