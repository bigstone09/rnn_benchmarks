# modified version of
# https://github.com/dmlc/mxnet/blob/master/example/rnn/lstm.py

import sys
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math
LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)


def lstm_unroll(num_lstm_layer, seq_len, input_size,
                num_hidden, num_embed, num_label, dropout=0.):
    """unrolled lstm network"""
    # initialize the parameter symbols
    embed_weight=mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight = mx.sym.Variable("l%d_i2h_weight" % i),
                                      i2h_bias = mx.sym.Variable("l%d_i2h_bias" % i),
                                      h2h_weight = mx.sym.Variable("l%d_h2h_weight" % i),
                                      h2h_bias = mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    label = mx.sym.Variable("label")
    last_hidden = []
    for seqidx in range(seq_len):
        # embeding layer
        data = mx.sym.Variable("t%d_data" % seqidx)

        hidden = mx.sym.Embedding(data=data, weight=embed_weight,
                                  input_dim=input_size,
                                  output_dim=num_embed,
                                  name="t%d_embed" % seqidx)
        # stack LSTM
        for i in range(num_lstm_layer):
            if i==0:
                dp=0.
            else:
                dp = dropout
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i, dropout=dp)
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        last_hidden.append(hidden)
    concat = mx.sym.Concat(*last_hidden, dim = 0)
    fc = mx.sym.FullyConnected(data=concat,
                               weight=cls_weight,
                               bias=cls_bias,
                               num_hidden=num_label)
    sm = mx.sym.SoftmaxOutput(data=fc, label=label, name="sm")
    out_prob = [sm]
    for i in range(num_lstm_layer):
        state = last_states[i]
        state = LSTMState(c=mx.sym.BlockGrad(state.c, name="l%d_last_c" % i),
                          h=mx.sym.BlockGrad(state.h, name="l%d_last_h" % i))
        last_states[i] = state

    unpack_c = [state.c for state in last_states]
    unpack_h = [state.h for state in last_states]
    list_all = out_prob + unpack_c + unpack_h
    return mx.sym.Group(list_all)


def is_param_name(name):
    return name.endswith("weight") or name.endswith("bias") or\
        name.endswith("gamma") or name.endswith("beta")


def setup_rnn_model(ctx,
                    num_lstm_layer, seq_len,
                    num_hidden, num_embed, num_label,
                    batch_size, input_size,
                    initializer, dropout=0.):
    """set up rnn model with lstm cells"""
    rnn_sym = lstm_unroll(num_lstm_layer=num_lstm_layer,
                          num_hidden=num_hidden,
                          seq_len=seq_len,
                          input_size=input_size,
                          num_embed=num_embed,
                          num_label=num_label,
                          dropout=dropout)
    arg_names = rnn_sym.list_arguments()

    input_shapes = {}
    for name in arg_names:
        if name.endswith("init_c") or name.endswith("init_h"):
            input_shapes[name] = (batch_size, num_hidden)
        elif name.endswith("data"):
            input_shapes[name] = (batch_size, )
        else:
            pass

    arg_shape, out_shape, aux_shape = rnn_sym.infer_shape(**input_shapes)
    arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape]
    args_grad = {}
    for shape, name in zip(arg_shape, arg_names):
        if is_param_name(name):
            args_grad[name] = mx.nd.zeros(shape, ctx)

    rnn_exec = rnn_sym.bind(ctx=ctx, args=arg_arrays,
                            args_grad=args_grad,
                            grad_req="add")
    param_blocks = []
    arg_dict = dict(zip(arg_names, rnn_exec.arg_arrays))
    for i, name in enumerate(arg_names):
        if is_param_name(name):
            initializer(name, arg_dict[name])

            param_blocks.append((i, arg_dict[name], args_grad[name], name))
        else:
            assert name not in args_grad
    out_dict = dict(zip(rnn_sym.list_outputs(), rnn_exec.outputs))

    init_states = [LSTMState(c=arg_dict["l%d_init_c" % i],
                             h=arg_dict["l%d_init_h" % i]) for i in range(num_lstm_layer)]
    seq_labels = rnn_exec.arg_dict["label"]
    seq_data = [rnn_exec.arg_dict["t%d_data" % i] for i in range(seq_len)]
    last_states = [LSTMState(c=out_dict["l%d_last_c_output" % i],
                             h=out_dict["l%d_last_h_output" % i]) for i in range(num_lstm_layer)]
    seq_outputs = out_dict["sm_output"]

    return LSTMModel(rnn_exec=rnn_exec, symbol=rnn_sym,
                     init_states=init_states, last_states=last_states,
                     seq_data=seq_data, seq_labels=seq_labels, seq_outputs=seq_outputs,
                     param_blocks=param_blocks)



def set_rnn_inputs(m, X, begin):
    seq_len = len(m.seq_data)
    batch_size = m.seq_data[0].shape[0]
    for seqidx in range(seq_len):
        idx = (begin + seqidx) % X.shape[0]
        next_idx = (begin + seqidx + 1) % X.shape[0]
        x = X[idx, :]
        y = X[next_idx, :]
        mx.nd.array(x).copyto(m.seq_data[seqidx])
        m.seq_labels[seqidx*batch_size : seqidx*batch_size+batch_size] = y

def calc_nll(seq_label_probs, X, begin):
    nll = -np.sum(np.log(seq_label_probs.asnumpy())) / len(X[0,:])
    return nll

if __name__ == "__main__":
    n_hidden = 128
    #n_hidden = 256
    #n_hidden = 512
    #n_hidden = 1024
    model = setup_rnn_model(mx.gpu(0),
                                 num_lstm_layer=1,
                                 seq_len=64,
                                 num_hidden=n_hidden,
                                 num_embed=n_hidden,
                                 num_label=5,
                                 batch_size=128,
                                 input_size=10000,
                                 initializer=mx.initializer.Uniform(0.1))
    def step():
        model.rnn_exec.forward()
        model.rnn_exec.backward()
        for p,g in zip(model.rnn_exec.arg_arrays, model.rnn_exec.grad_arrays):
            if g:
                p[:] = p-g*0.001

    step()
    step()
    mx.nd.waitall()
    t_start = time.time()
    for i in range(50):
        print i
        step()
    mx.nd.waitall()
    tdiff = time.time() - t_start
    print "took", tdiff
    print tdiff / 50.

