# LSTM benchmarks

## Disclaimer
These tests were made *very* quickly for me to get a rough sense of how
the different frameworks stack up against each other for LSTM and as such are
not 100% accurate.

In addition, I am new to a number of these frameworks and am sure I have
made catastrophic mistakes somewhere. I would strongly recommend running
your own tests before making a choice between framework.

In addition to this, zero work was put into optimizing the different
versions. Most of them were simply pulled from someplace on the Internet (see top line comment) then modified / trimmed down.

If you find an issue or want to add more frameworks, feel free to open an issue or send a Pull Request.

## Target Model
Each implementation is modeled off of an Embedding table -> LSTM ->
slice off last element -> Fully Connected -> Softmax -> categorical
cross entropy.

The models have 10k word vocab, with the same size hidden and embedding table dimensions.
All classification is 5 way.

The numbers represent 1 forward pass, 1 backward pass, and a SGD
parameter update. They are calculated as an average of 50 batches.

The following numbers are on a single 980. All numbers are seconds per batch.

| framework | hidden-128 | hidden-256| hidden-512| hidden-1024| hidden-2048|
| --------- |:----------:| ---------:|-------:| ----:| --:|
| Theano    | 0.0271 |  0.0400 | 0.0758 | 0.1684 | 0.5267
| Tensorflow(7.1) | 0.0729 | 0.0751 | 0.0899 | 0.1865| out of mem|
| mxnet | 0.2672| 0.4902 | 0.9500 | Out of mem| didn't run|
| torch (element-research/rnn)| 0.0742 |0.0796 | 0.0947| 0.2043| out of mem|
| keras (theano) | 0.0703 | 0.0974 | .1456 | recursion depth| didn't run|
| keras (tensorflow) | 0.1188 | .1438 | .2146 | .4433| didn't run|
