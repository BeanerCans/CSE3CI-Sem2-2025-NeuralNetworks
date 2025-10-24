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
num_classes = 10
y_train_cat = utils.to_categorical(y_train, num_classes)
y_test_cat = utils.to_categorical(y_test, num_classes)

print("Train shape:", x_train.shape, "Test shape:", x_test.shape)

# Helper: plotting utilities
def plot_history(history, title_prefix=""):

    # accuracy
    plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train_acc')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title(title_prefix + " Accuracy")
    plt.legend()

    # loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train_loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(title_prefix + " Loss")
    plt.legend()
    plt.show()

def evaluate_and_print(model, x_test, y_test_cat):
    loss, acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"Test Loss: {loss:.4f} | Test Accuracy: {acc:.4f}")
    return loss, acc

