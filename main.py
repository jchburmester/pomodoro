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

# Start a parser with the arguments
parser = argparse.ArgumentParser(description='ResNet50')

# Preprocessing arguments
parser.add_argument('--augmentation', type=bool, default=False, help='Augmentation methods')
parser.add_argument('--normalization', type=bool, default=False, help='Normalization methods')
parser.add_argument('--scaling', type=bool, default=False, help='Scaling methods')
parser.add_argument('--new_partitioning', type=bool, default=False, help='Data splitting methods')
parser.add_argument('--higher_precision_casting', type=bool, default=False, help='Casting data to float64')

# Training arguments
parser.add_argument('--optimizer', type=str, default='SGD', help='name of optimizer') # Others: Adam, RMSprop
parser.add_argument('--batch_size', type=int, default=32, help='Batch size') # Others: 64, 128
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate') # Others: 0.01, 0.0001
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum') # Others: 0.5, 0.99
parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay') # Others: 0.001, 0.00001
parser.add_argument('--global_quantization', type=bool, default=False, help='Global quantization') # Others: True
parser.add_argument('--model_quantization', type=bool, default=False, help='Local quantization') # Others: True
parser.add_argument('--jit_compilation', type=bool, default=False, help='JIT compilation') # Others: True

# Postprocessing (inferece) arguments
parser.add_argument('--post_quantization', type=bool, default=False, help='Post traingin quantization')
parser.add_argument('--weight_clustering', type=bool, default=False, help='Weight clustering')
parser.add_argument('--weight_pruning', type=bool, default=False, help='Weight pruning')
parser.add_argument('--tf_lite_conversion', type=bool, default=False, help='TF Lite conversion')

# Parse the arguments
args = parser.parse_args()

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
    train_size = data.take(int(0.01 * len(data)))
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

# Pick optimizer
if args.optimizer == 'SGD':
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        nesterov=True,
    )
elif args.optimizer == 'Adam':
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
elif args.optimizer == 'RMSprop':
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

combined_model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
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