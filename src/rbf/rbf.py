#%%
import numpy as np
import os
import random as python_random
from keras.datasets import boston_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import tensorflow as tf
import tensorflow_addons as tfa
from keras.metrics import RootMeanSquaredError
from tensorflow_addons.metrics import RSquare
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer
from scipy.spatial.distance import pdist
from keras.initializers import Initializer
from sklearn.cluster import KMeans
import tensorflow.keras.backend as K



def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_CUDNN_DETERMINISTIC"] = str(seed)

    # source: https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    # source: https://github.com/keras-team/keras/issues/2743
    np.random.seed(seed)
    python_random.seed(seed)
    tf.random.set_seed(seed)

#%%

# Helpers
# plot learning curves
def plot_history(history):
    epochs = len(history.history["accuracy"])
    x = np.arange(1, epochs + 1)
    plt.figure(constrained_layout=True)
    plt.subplot(211)
    plt.plot(x, history.history["r_square"])
    plt.plot(x, history.history["val_r_square"], color="green")
    plt.legend(["train", "validation"], loc="upper left")

    plt.subplot(211)
    plt.plot(x, history.history["r_square"])
    plt.plot(x, history.history["val_r_square"], color="green")
    plt.legend(["train", "validation"], loc="upper left")


    plt.subplot(212)
    plt.plot(x, history.history["loss"])
    plt.plot(x, history.history["val_loss"], color="green")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(["train", "validation"], loc="upper right")


#%%

# the feature B isn't included to avoid ethnical problems
# source: https://keras.io/api/datasets/boston_housing/
(train_x, train_y), (test_x, test_y) = boston_housing.load_data(test_split=0.25)

num_samples, num_features = train_x.shape

# z-score normalization
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)


#%%

class InitCentersKMeans(Initializer):
    def __init__(self, X, max_iter=100):
        self.X = X
        self.max_iter = max_iter

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]

        n_centers = shape[0]
        km = KMeans(n_clusters=n_centers, max_iter=self.max_iter, verbose=0)
        km.fit(self.X)
        return km.cluster_centers_

class RBFWithKMeansLayer(Layer):
    def __init__(self, output_dim, train_data, **kwargs):
        self.output_dim = output_dim
        self.initializer = InitCentersKMeans(train_data)
        super(RBFWithKMeansLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=False)
        max_dist = max(pdist(self.centers))
        sigma = max_dist / np.sqrt(2 * self.output_dim)
        self.gamma = 1/(2*(sigma**2))
        super(RBFWithKMeansLayer, self).build(input_shape)

    def call(self, x):
        C = K.expand_dims(self.centers)
        H = K.transpose(C-K.transpose(x))
        return K.exp(-self.gamma * K.sum(H**2, axis=1))


def rbf_model(num_neurons):
    model = Sequential()
    model.add(RBFWithKMeansLayer(num_neurons, train_data=train_x, input_shape=(num_features,)))
    model.add(Dense(128))
    model.add(Dense(1))
    model.compile(optimizer=SGD(learning_rate=0.001), 
                  loss="mse", 
                  metrics=["accuracy", RootMeanSquaredError(), RSquare()])
    return model

def fit(model):
    return model.fit(train_x, train_y, batch_size=32, epochs=100, validation_split=0.2)


hidden_neurons = [int(0.1*num_samples), int(0.5*num_samples), int(0.9*num_samples)]
for neurons in hidden_neurons:
    set_seed(1)
    print("Neurons", neurons)
    model = rbf_model(neurons)
    plot_history(fit(model))
    print("Evaluation", model.evaluate(test_x, test_y))
