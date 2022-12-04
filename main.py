""" 
Main file for the project
Pomodoro, 3.12.2022
"""

import tensorflow as tf
import numpy as np
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from preprocessing import Preprocessing
from resnet50 import ResNet50

# Start a parser with the arguments
parser = argparse.ArgumentParser(description='ResNet50')

# Preprocessing arguments
parser.add_argument('--augmentation', type=bool, default=False, help='Augmentation methods') # All image augmentation
parser.add_argument('--normalization', type=bool, default=False, help='Normalization methods')
parser.add_argument('--standardization', type=bool, default=False, help='Standardization methods')
parser.add_argument('--90/5/5-partitioning', type=bool, default=False, help='Data splitting methods')

# Training arguments
parser.add_argument('--optimizer', type=str, default='SGD', help='name of optimizer')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
parser.add_argument('--global_quantization', type=bool, default=False, help='Global quantization')
parser.add_argument('--model_quantization', type=bool, default=False, help='Local quantization')
parser.add_argument('--jit_compilation', type=bool, default=False, help='JIT compilation')

# Postprocessing (inferece) arguments
parser.add_argument('--post_quantization', type=bool, default=False, help='Post traingin quantization')
parser.add_argument('--weight_clustering', type=bool, default=False, help='Weight clustering')
parser.add_argument('--weight_pruning', type=bool, default=False, help='Weight pruning')
parser.add_argument('--tf_lite_conversion', type=bool, default=False, help='TF Lite conversion')

# Parse the arguments
args = parser.parse_args()

# Initialize preprocessing
preprocessing_layer = Preprocessing(args.augmentation, 
                                    args.normalization, 
                                    args.standardization, 
                                    args.90/5/5-partitioning)

model = ResNet50(
    args.optimizer,
    args.batch_size,
    args.learning_rate,
    args.momentum,
    args.weight_decay,
    args.global_quantization,
    args.model_quantization,
    args.jit_compilation,
)

# Initialize model, stack the preprocessing layer and the model
model = tf.keras.Sequential([
    preprocessing_layer,
    model
])

# Data loading
tf.keras.utils.image_dataset_from_directory(
    'data/',
    labels='inferred',
    label_mode='categorical',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(224, 224),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
)