""" 
Main file for the project
Pomodoro, 3.12.2022
"""
SEED = 22

import tensorflow as tf
import numpy as np
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from preprocessing import Preprocessing
from resnet50 import ResNet50
from custom_callback import CSVLogger
from combi import base_line, random_combi


# Start a parser with the arguments
parser = argparse.ArgumentParser(description='Configuration for the training of the model')

# Parsing arguments if needed for the shell pipeline
parser.add_argument('--baseline_training', action='store_true', help='argument for training the model with no or the most basic parameters')
parser.add_argument('--n', type=int, default=1, help='number of training runs')

# Parse the arguments
args = parser.parse_args()

parameters = random_combi()

if args.baseline_training:
    parameters = base_line()

# Load the CIFAR100 dataset
data = tf.keras.datasets.cifar100.load_data()

X = data[0][0]
y = data[0][1]

# Convert to tf.data.Dataset
data = tf.data.Dataset.from_tensor_slices((X, y))

# Shuffle the data
data = data.shuffle(len(data), seed=SEED)

#####################################################################################
############################ Precision casting ######################################
#####################################################################################

# Cast the data to higher precision if needed
if parameters['higher_precision_casting'] is not 'None':
    data = data.map(lambda x, y: (tf.cast(x, parameters['higher_precision_casting']), y))

#####################################################################################
############################ Batch size #############################################
#####################################################################################

# batch the data
data = data.batch(parameters['batch_size'])

#####################################################################################
############################ Preprocessing ##########################################
#####################################################################################

# Split the data according the argument partitioning, either 90/5/5 or 70/15/15
if parameters['preprocessing'] is 'new_partitioning':
    train_size = data.take(int(0.9 * len(data)))
    val_size = data.skip(int(0.9 * len(data))).take(int(0.05 * len(data)))
    test_size = data.skip(int(0.95 * len(data))).take(int(0.05 * len(data)))
else:
    train_size = data.take(int(0.07 * len(data)))
    val_size = data.skip(int(0.7 * len(data))).take(int(0.15 * len(data)))
    test_size = data.skip(int(0.85 * len(data))).take(int(0.15 * len(data)))

# Initialize preprocessing
preprocessing_layer = Preprocessing(SEED, 
                                    data.as_numpy_iterator().next()[0].shape,
                                    parameters['preprocessing'])

#####################################################################################
############################ Model building #########################################
#####################################################################################

# Initialize model
model = ResNet50(
    len(os.listdir('data/train')),
    input_shape=(256, 256, 3),
    jit_compilation=args.jit_compilation,
)

# Initialize model, stack the preprocessing layer and the model
combined_model = tf.keras.Sequential([
    preprocessing_layer,
    model
])

#####################################################################################
############################ Learning Rate ##########################################
#####################################################################################

# Get learning rate
learning_rate = parameters['lr']

#####################################################################################
############################ Learning Rate Schedule #################################
#####################################################################################

# Pick learning rate schedule
if parameters['lr_schedule'] == 'constant':
    learning_rate_schedule = tf.keras.optimizers.schedules.Constant(learning_rate)
elif parameters['lr_schedule'] == 'exponential':
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True)
elif parameters['lr_schedule'] == 'polynomial':
    learning_rate_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        learning_rate,
        decay_steps=10000,
        end_learning_rate=0.0001,
        power=1.0,
        cycle=False)
elif parameters['lr_schedule'] == 'cosine':
    learning_rate_schedule = tf.keras.optimizers.schedules.CosineDecay(
        learning_rate,
        decay_steps=10000,
        alpha=0.0)

#####################################################################################
############################ Optimizer Momentum #####################################
#####################################################################################

# Get optimizer momentum
optimizer_momentum = parameters['optimizer_momentum']

#####################################################################################
############################ Optimizer ##############################################
#####################################################################################

# Pick optimizer
if parameters['optimizer'] == 'SGD':
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate_schedule,
        momentum=optimizer_momentum)
elif parameters['optimizer'] == 'RMSprop':
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=learning_rate_schedule,
        momentum=optimizer_momentum)
elif parameters['optimizer'] == 'Adam':
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_schedule)
elif parameters['optimizer'] == 'AdamW':
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate_schedule,
        weight_decay=0.0001)

#####################################################################################
############################ Training ###############################################
#####################################################################################

combined_model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy'],
    jit_compile=args.jit_compilation,
)

# Train the model
combined_model.fit(
    train_size,
    validation_data=val_size,
    epochs=1
)

# Quantize the model
if args.global_quantization:
    converter = tf.lite.TFLiteConverter.from_keras_model(combined_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    open('resnet50_quant.tflite', 'wb').write(tflite_quant_model)

# Quantize the model
if args.model_quantization:
    converter = tf.lite.TFLiteConverter.from_keras_model(combined_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_quant_model = converter.convert()
    open('resnet50_quant.tflite', 'wb').write(tflite_quant_model)