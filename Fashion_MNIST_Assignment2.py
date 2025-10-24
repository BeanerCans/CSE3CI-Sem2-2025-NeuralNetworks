# Requirements: tensorflow (2.x), matplotlib, numpy

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, utils, optimizers
import matplotlib.pyplot as plt
import os
import datetime

# 1. Load and preprocess data
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalise and reshape
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

# One-hot encode labels