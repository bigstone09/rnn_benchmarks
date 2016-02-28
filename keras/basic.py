# modified from
# https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.datasets import imdb
import time

max_features = 10000
maxlen = 64 # cut texts after this number of words (among top max_features most common words)
batch_size = 128
#n_hidden = 128
# n_hidden = 256
#n_hidden = 512
#n_hidden = 1024
#n_hidden = 1024

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features,
                                                      test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, n_hidden, input_length=maxlen, dropout=0.0))
model.add(LSTM(n_hidden, dropout_W=0.0, dropout_U=0.0))  # try using a GRU instead, for fun
#model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='sgd')


print("Train...")
X_train = X_train[0:128*50]
y_train = y_train[0:128*50]
y_train = np_utils.to_categorical(y_train, 5)
t_start = time.time()
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1)
tdiff = time.time() - t_start
print tdiff
print tdiff / 50.
