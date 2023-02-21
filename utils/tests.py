""" 
Testing file for the new callback
Pomodoro, 4.12.2022
"""
import sys
sys.path.append('')
import os
import numpy as np
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from models.resnet50 import load_resnet50
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from utils.callback import SMICallback

# Load the mnist dataset
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Only use 1% of the data
x_train = x_train[:800]
y_train = y_train[:800]

# You have to expand dims to meet the minimum requirements of the model
x_train = np.array([np.array(Image.fromarray(x).resize((32, 32))) for x in x_train])
x_test = np.array([np.array(Image.fromarray(x).resize((32, 32))) for x in x_test])

# Normalize the dataset
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape the dataset
x_train = x_train.reshape(-1, 32, 32, 1)
x_test = x_test.reshape(-1, 32, 32, 1)

# Create the model
model = load_resnet50(classes=10, input_shape=(32, 32, 1))

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train the model
model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_data=(x_test, y_test),
    callbacks=[SMICallback()]
)