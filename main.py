""" 
Main file for the project
Pomodoro, 3.12.2022
"""

SEED = 22

import tensorflow as tf
# import tensorflow_addons as tfa We are using the old AdamW 
import tensorflow_model_optimization as tfmot
import numpy as np
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils.subfolder_creation import create_subfolder
from utils.preprocessing import Preprocessing
from utils.custom_callback import CSVLogger
from utils.config_creator import base_line, random_config
from utils.mixup import mixup
from utils.cutmix import cutmix
from models.resnet50 import load_resnet50
from models.convnextv1 import load_convnextv1

# Start a parser with the arguments
parser = argparse.ArgumentParser(description='Configuration for the training of the model')

# Parsing arguments if needed for the shell pipeline
parser.add_argument('--baseline', action='store_true', help='argument for training the model with no or the most basic parameters')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
parser.add_argument('--model', type=str, default='resnet50', help='model to train; options: resnet50, convnextv1')

# Parse the arguments
args = parser.parse_args()

#####################################################################################
############################ Initialise Parameters ##################################
#####################################################################################

parameters = random_config()

if args.baseline:
    parameters = base_line()

#####################################################################################
############################ Create Folders & Parameter File ########################
#####################################################################################

current_dir = create_subfolder()

with open(os.path.join('runs', current_dir, 'parameters.txt'), 'w') as f:
    # store model name
    f.write(str(args.model)+'\n')
    # store parameters
    for key, value in parameters.items():
        f.write(str(key)+': '+str(value)+'\n')

#####################################################################################
############################ Load data ##############################################
#####################################################################################

(x_train, y_train), (_, _) = tf.keras.datasets.cifar100.load_data()

if parameters['preprocessing'] == 'normalization':
    x_train = x_train.astype('float32') / 255.
elif parameters['preprocessing'] == 'standardization':
    x_train = x_train.astype('float32')
    x_train = (x_train - np.mean(x_train, axis=(0, 1, 2))) / (np.std(x_train, axis=(0, 1, 2)))
elif parameters['preprocessing'] == 'minmax':
    x_train = x_train.astype('float32')
    x_train = (x_train - np.min(x_train, axis=(0, 1, 2))) / (np.max(x_train, axis=(0, 1, 2)) - np.min(x_train, axis=(0, 1, 2)))

x_train = tf.reshape(x_train, (-1, 32, 32, 3))
y_train = tf.keras.utils.to_categorical(y_train, 100)

if parameters['augmentation'] == 'mixup':
    mix_ds1 = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024).batch(1)
    mix_ds2 = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024).batch(1)
    mix_ds = tf.data.Dataset.zip((mix_ds1, mix_ds2))
    data = mix_ds.map(lambda ds1, ds2: mixup(ds1, ds2, 0.2), num_parallel_calls=tf.data.experimental.AUTOTUNE)
elif parameters['augmentation'] == 'cutmix':
    mix_ds1 = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024).batch(1)
    mix_ds2 = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024).batch(1)
    mix_ds = tf.data.Dataset.zip((mix_ds1, mix_ds2))
    data = mix_ds.map(lambda ds1, ds2: cutmix(ds1, ds2), num_parallel_calls=tf.data.experimental.AUTOTUNE)
elif parameters['augmentation'] == 'random':
    data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    preprocessing_layer = Preprocessing(SEED, data.as_numpy_iterator().next()[0].shape, random=True)
elif parameters['augmentation'] == 'None': 
    data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    preprocessing_layer = Preprocessing(SEED, data.as_numpy_iterator().next()[0].shape, random=False)

# Shuffle the data
data = data.shuffle(1024, seed=SEED)

#####################################################################################
############################ Precision casting ######################################
#####################################################################################

if parameters['precision'] == 'global_policy_float16':
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
else:
    data = data.map(lambda x, y: (tf.cast(x, tf.dtypes.as_dtype(str(parameters['precision']))), y))

#####################################################################################
############################ Batch size #############################################
#####################################################################################

# batch the data
data = data.batch(int(parameters['batch_size']))

#####################################################################################
############################ Partitioning ###########################################
#####################################################################################

split = [round(float(x) * 0.01, 2) for x in parameters['partitioning'].split('-')]

# Split the data according the argument partitioning
train_size = int(split[0] * len(data))
train_ds = data.take(train_size)
val_ds = data.skip(train_size).take(int(split[1] * len(data)))
test_ds = data.skip(int(round(split[0] + split[1],2) * len(data))).take(int(split[1] * len(data)))

# Print the size of the data
print('\n')
print('Size of the training set: ', len(train_ds))
print('Size of the validation set: ', len(val_ds))
print('Size of the test set: ', len(test_ds))

#####################################################################################
############################ Model building #########################################
#####################################################################################

if args.model == 'resnet50':
    model = load_resnet50(classes=100, input_shape=data.as_numpy_iterator().next()[0].shape, weights=None)

elif args.model == 'convnextv1':
    model = load_convnextv1(classes=100, input_shape=data.as_numpy_iterator().next()[0].shape, weights=None)

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
    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=learning_rate_schedule,
        weight_decay=parameters['weight_decay'])

#####################################################################################
############################ Training ###############################################
#####################################################################################

if parameters['quantization'] == 'pre':
    quantize_model = tfmot.quantization.keras.quantize_model
    combined_model = quantize_model(combined_model)

if parameters['internal_optimization'] == 'jit_compilation':
    combined_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'],
        jit_compile=True
    )
else:
    combined_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'],
        jit_compile=False
    )

# Train the model
combined_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs
)

#####################################################################################
############################ Quantization ###########################################
#####################################################################################

# Global quantization
if parameters['quantization'] == 'post_weights':
    converter = tf.lite.TFLiteConverter.from_keras_model(combined_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    combined_model = tf.keras.models.load_model(tflite_quant_model)

# Model quantization
if parameters['quantization'] == 'post_weights_and_activations':
    def representative_dataset_gen():
        for input_value, _ in train_ds.take(100):
            yield [input_value]

    converter = tf.lite.TFLiteConverter.from_keras_model(combined_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    tflite_quant_model = converter.convert()
    combined_model = tf.keras.models.load_model(tflite_quant_model)

#####################################################################################
############################ Pruning ################################################
#####################################################################################

if parameters['internal_optimizations'] == 'weight_pruning':
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    epochs = 2
    end_step = np.ceil(train_size / parameters['batch_size']).astype(np.int32) * epochs

    # hyperparameters as in the tf documentation
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity = 0.50,
        final_sparsity = 0.80,              
        begin_step = 0,                                                               
        end_step = end_step)
    }
    combined_model = prune_low_magnitude(combined_model, **pruning_params)
    
    combined_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

#####################################################################################
############################ Weight Clustering ######################################
#####################################################################################

if parameters['internal_optimizations'] == 'weight_clustering':
    cluster_weights = tfmot.clustering.keras.cluster_weights
    CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

    # hyperparameters as in the tf documentation
    clustering_params = {
        'number_of_clusters': 16,
        'cluster_centroids_init': CentroidInitialization.LINEAR
    }
    combined_model = cluster_weights(combined_model, **clustering_params)

    # Use smaller learning rate for fine-tuning clustered model
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

    combined_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=opt,
        metrics=['accuracy']
    )

#####################################################################################
############################ Model Fit ##############################################
#####################################################################################

csv_logger = CSVLogger(os.path.join(current_dir, 'training.log'))

combined_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
    callbacks=[csv_logger]
)

# save_metric(model.evaluate(test_ds))

# save metrics to csv: run#, model_name, preprocessing, augmentation, precision, batch, partitioning, 
# lr, lr_schedule, optimizer, optimizer_momentum, quantization, internal_optimizations, train_accuracy, 
# test_accuracy, loss, #parameters, time_elapsed

# save metrics to yaml (in run folder): run#, model_name, preprocessing, augmentation, precision, batch, 
# partitioning, lr, lr_schedule, optimizer, optimizer_momentum, quantization, internal_optimizations,  
# train_accuracy, test_accuracy, loss, #parameters, time_elapsed