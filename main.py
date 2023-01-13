""" 
Main file for the project
Pomodoro, 3.12.2022
"""
SEED = 22
DIR = 'data/train/'

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
parser = argparse.ArgumentParser(description='ResNet50')

# Parsing arguments if needed for the shell pipeline
parser.add_argument('--baseline_training', action='store_true', help='argument for training the model with no or the most basic parameters')
parser.add_argument('--random_training', action='store_true', help='argument for training the model with random parameters')

# Parse the arguments
args = parser.parse_args()

# Get the parameters for the model
if args.baseline_training:
    parameters = base_line()

if args.random_training:
    parameters = random_combi()

# Iterate over parameter dictionary and add the parameters to the model
for key, value in parameters.items():
    if key == 'preprocessing':
        args.preprocessing = value
    if key == 'higher_precision_casting':
        args.higher_precision_casting = value
    if key == 'batch_size':
        args.batch_size = value
    if key == 'lr':
        args.learning_rate = value
    if key == 'lr_schedule':
        args.lr_schedule = value
    if key == 'optimizer':
        args.optimizer = value
    if key == 'optimizer_momentum':
        args.optimizer_momentum = value
    if key == 'weight_decay':
        args.weight_decay = value
    if key == 'quantization':
        args.quantization = value
    if key == 'postprocessing':
        args.postprocessing = value

#####################################################################################
############################ Done until this part ###################################
#####################################################################################

# Data loading
data = tf.keras.utils.image_dataset_from_directory(
    DIR,
    labels='inferred',
    label_mode='categorical',
    class_names=None,
    color_mode='rgb',
    batch_size=args.batch_size,
    image_size=(256, 256),
    shuffle=True,
    seed=SEED,
    interpolation='bilinear',
)


# Convert to float32
if args.higher_precision_casting:
    data = data.map(lambda x, y: (tf.cast(x, tf.float64), y))

# Split the data according the argument partitioning, either 90/5/5 or 70/15/15
if args.new_partitioning:
    train_size = data.take(int(0.9 * len(data)))
    val_size = data.skip(int(0.9 * len(data))).take(int(0.05 * len(data)))
    test_size = data.skip(int(0.95 * len(data))).take(int(0.05 * len(data)))
else:
    train_size = data.take(int(0.07 * len(data)))
    val_size = data.skip(int(0.7 * len(data))).take(int(0.15 * len(data)))
    test_size = data.skip(int(0.85 * len(data))).take(int(0.15 * len(data)))

# Initialize preprocessing
preprocessing_layer = Preprocessing(SEED, 
                                    (256,256,3),
                                    args.augmentation, 
                                    args.normalization, 
                                    args.scaling)

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

# Get learning rate
if args.learning_rate:
    learning_rate_v = 0.01

# Pick optimizer
if args.optimizer:
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate_v,
        momentum=args.momentum,
        nesterov=True,
    )
elif args.optimizer == 'Adam': # not needed at the moment
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_v,
        weight_decay=args.weight_decay,
    )
elif args.optimizer == 'RMSprop': # not needed at the moment
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=learning_rate_v,
        weight_decay=args.weight_decay,
    )

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