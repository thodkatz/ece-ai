# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
# set the seed to get reproducible results
import os
seed=2
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_CUDNN_DETERMINISTIC']= str(seed)

import numpy as np
import tensorflow as tf
import random as python_random
# source: https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
# source: https://github.com/keras-team/keras/issues/2743
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)

import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import mnist
from keras import Sequential
from keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#%%
# load and normalize dataset
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# vectorize image row-wise
# shape = (num_samples, num_features)
train_x = train_x.reshape(train_x.shape[0], -1)
test_x = test_x.reshape(test_x.shape[0], -1)

num_samples_training = train_x.shape[0]
num_features = train_x.shape[1]
num_classes = 10

# memory efficient
train_x = train_x.astype('float32')
test_x = test_x.astype('float32')

# %%
# min max normalization
train_x = train_x / 255
test_x = test_x / 255
# plt.hist(train_x[0,:])


# %%
# z score
train_x = StandardScaler().fit_transform(train_x)
test_x = StandardScaler().fit_transform(test_x)
plt.hist(train_x[0, :])

# %%
# min-max sklearn
train_x = MinMaxScaler().fit_transform(train_x)
test_x = MinMaxScaler().fit_transform(test_x)
plt.hist(train_x[0, :])

# %%
# L2 normalization
train_x = keras.utils.normalize(train_x, axis=1)
test_x = keras.utils.normalize(test_x, axis=1)
plt.hist(train_x[0, :])

# %%
# build the neural network
hidden_layer_nodes1 = 128
hidden_layer_nodes2 = 256
model = Sequential(
    [Flatten(input_shape=(num_features,)),
     Dense(hidden_layer_nodes1, activation='relu'),
     Dense(hidden_layer_nodes2, activation='relu'),
     Dense(num_classes, activation='softmax')]
)

# %%
# sgd
# or but requires one-hot encoding first (to_categorical)
train_y = keras.utils.to_categorical(train_y, num_classes)
test_y = keras.utils.to_categorical(test_y, num_classes)
model.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_x, train_y, batch_size = num_samples_training, epochs=5, validation_data=(test_x, test_y),shuffle=False)

# %%
# rmsprop
model.compile(optimizer=RMSprop(learning_rate=0.001,rho=0.9),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#print(model.summary())
#print(model.weights)
for layer in model.layers:
    print(layer.get_weights())
model.fit(train_x, train_y, batch_size = num_samples_training, epochs=5, validation_data=(test_x, test_y),shuffle=False)

#%%
# sgd
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=10,batch_size = num_samples_training, validation_data=(test_x, test_y),shuffle=False)

# %%
model.evaluate(test_x, test_y)
