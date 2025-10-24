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

# 2. Task 1: Models without convolution (Dense networks)

# Baseline dense: Flatten -> Dense(128) -> Dense(10)
def build_dense_baseline():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Improved dense: Flatten -> Dense(512) -> Dense(256) -> Dense(10)
def build_dense_improved():
    model = models.Sequential([
        layers.input(shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Compile & train helper
def compile_and_train(model, optimizer, epochs=10, batch_size=128, use_val=False):
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if use_val:
        history = model.fit(x_train, y_train_cat, validation_data=(x_test, y_test_cat),
                            epochs=epochs, batch_size=batch_size, verbose=2)
    else:
        history = model.fit(x_train, y_train_cat, epochs=epochs, batch_size=batch_size, verbose=2)
    return history

# Run dense baseline
dense1 = build_dense_baseline()
print("Dense baseline summary:")
dense1.summary()
hist_dense1 = compile_and_train(dense1, optimizer='adam', epochs=8)
plot_history(hist_dense1, "Dense Baseline")
loss_d1, acc_d1 = evaluate_and_print(dense1, x_test, y_test_cat)

# Run improved dense
dense2 = build_dense_improved()
print("Dense improved summary:")
dense2.summary()
hist_dense2 = compile_and_train(dense2, optimizer='adam', epochs=10)
plot_history(hist_dense2, "Dense Improved")
loss_d2, acc_d2 = evaluate_and_print(dense2, x_test, y_test_cat)
