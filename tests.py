""" 
Testing file for the new callback
Pomodoro, 4.12.2022
"""
from resnet50 import ResNet50
from custom_callback import CSVLogger
import tensorflow as tf

# Load the mnist dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Only use 1% of the data
x_train = x_train[:600]
y_train = y_train[:600]

# Normalize the dataset
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape the dataset
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Create the model
model = ResNet50(num_classes=10, input_shape=(28, 28, 1))

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model
model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_data=(x_test, y_test),
    callbacks=[CSVLogger('results.csv')]
)