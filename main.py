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
#import wandb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from preprocessing import Preprocessing
from resnet50 import ResNet50
from custom_callback import CSVLogger

# Start a parser with the arguments
parser = argparse.ArgumentParser(description='ResNet50')

# Preprocessing arguments
parser.add_argument('--augmentation', action='store_true', help='Augmentation methods')
parser.add_argument('--normalization', action='store true', help='Normalization methods')
parser.add_argument('--scaling', action='store true', help='Scaling methods')
parser.add_argument('--new_partitioning', action='store true', help='Data splitting methods')
parser.add_argument('--higher_precision_casting', action='store true', help='Casting data to float64')

# Training arguments
parser.add_argument('--optimizer', action='store true', help='name of optimizer') # Default: SGD, others: Adam, RMSprop
parser.add_argument('--batch_size', action='store true', help='Batch size') # Default: 32, others: 64, 128
parser.add_argument('--learning_rate', action='store true', help='Learning rate') # Default: 0.001, others: 0.01, 0.0001
parser.add_argument('--momentum', action='store true', help='Momentum') # Default: 0.9, others: 0.5, 0.99
parser.add_argument('--weight_decay', action='store true', help='Weight decay') # Default: 0.0001, others: 0.001, 0.00001
parser.add_argument('--global_quantization', action='store true', help='Global quantization')
parser.add_argument('--model_quantization', action='store true', help='Local quantization')
parser.add_argument('--jit_compilation', action='store true', help='JIT compilation')

# Postprocessing (inferece) arguments
parser.add_argument('--post_quantization', type=bool, default=False, help='Post traingin quantization')
parser.add_argument('--weight_clustering', type=bool, default=False, help='Weight clustering')
parser.add_argument('--weight_pruning', type=bool, default=False, help='Weight pruning')
parser.add_argument('--tf_lite_conversion', type=bool, default=False, help='TF Lite conversion')

# Parse the arguments
args = parser.parse_args()
print(args)

learning_rate_v = 0.001 # Default learning rate

# Global quantization (float32, float64, float16... )
# Can also be the string 'mixed_float16' or 'mixed_bfloat16', 
# which causes the compute dtype to be float16 or bfloat16 and the variable dtype to be float32.
if args.global_quantization:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print('Global quantization is enabled')

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