# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
# set the seed to get reproducible results
from enum import Enum
from black import out
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from keras.layers import Flatten, Dense, Dropout
from keras import Sequential
from keras.initializers import RandomNormal
from keras.datasets import mnist
from keras.regularizers import l2, l1
import matplotlib.pyplot as plt
import random as python_random
import tensorflow as tf
import numpy as np
import os

seed = 1
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["TF_CUDNN_DETERMINISTIC"] = str(seed)

# source: https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
# source: https://github.com/keras-team/keras/issues/2743
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)


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
train_x = train_x.astype("float32")
test_x = test_x.astype("float32")

# min-max normalization
train_x = train_x / 255
test_x = test_x / 255

# build the neural network
hidden_layer_nodes1 = 128
hidden_layer_nodes2 = 256

# Notes:
# regularization can be used in the output layer too, although in most examples they don't include it
# dropout should not be used for input and output layers
def create_model(custom_weight_init=False, l2=False, l2_alpha=0.1, l1_dropout=False):
    if l2 and l1_dropout:
        print("Conflict in regularization l2-l1_dropout")
        exit(-1)
    hidden_layer_options = {}
    output_layer_options = {}
    if custom_weight_init:
        hidden_layer_options["kernel_initializer"] = RandomNormal(mean=10)
        output_layer_options["kernel_initializer"] = RandomNormal(mean=10)
    if l2:
        hidden_layer_options["kernel_regularizer"] = l2(l2_alpha)
    if l1_dropout:
        hidden_layer_options["kernel_regularizer"] = l1(0.001)

    model = Sequential()
    # 1st hidden layer
    model.add(
        Dense(
            hidden_layer_nodes1,
            input_shape=(num_features,),
            activation="relu",
            **hidden_layer_options
        )
    )

    if l1_dropout:
        # source: https://machinelearningmastery.com/how-to-reduce-overfitting-with-dropout-regularization-in-keras/
        model.add(Dropout(0.3))

    # 2nd hidden layer
    model.add(Dense(hidden_layer_nodes2, activation="relu", **hidden_layer_options))

    if l1_dropout:
        model.add(Dropout(0.3))

    # output
    model.add(Dense(num_classes, activation="softmax", **output_layer_options))

    return model


model = create_model(custom_weight_init=True)
# print(model.summary())


class Optimizer(Enum):
    RMSPROP = 1
    SGD = 2


def fitWrapper(batch_size, epochs, optimizer, rho):
    if optimizer == Optimizer.RMSPROP:
        model.compile(
            optimizer=RMSprop(learning_rate=0.001, rho=rho),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    elif optimizer == Optimizer.SGD:
        model.compile(
            optimizer="sgd",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    else:
        print("Not supported optimizer")
        exit(-1)

    model.fit(
        train_x,
        train_y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(test_x, test_y),
        validation_split=0.2,
        shuffle=False,
    )


batches = [1, 256, num_samples_training]
fitWrapper(num_samples_training, 10, Optimizer.RMSPROP, rho=0.9)

# test model with unseen data
model.evaluate(test_x, test_y)