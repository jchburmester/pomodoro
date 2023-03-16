""" 
Main file for the project
Pomodoro, 3.12.2022
"""

SEED = 22
DEBUG = False    

# External imports
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt
import numpy as np
import argparse
import yaml
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Internal imports
from utils.config_creator import base_line, random_config
from utils.subfolder_creation import create_subfolder
from utils.preprocessing import Preprocessing
from utils.mixup import mixup
from utils.cutmix import cutmix
from models.resnet50 import load_resnet50
from models.convnextv1 import load_convnextv1
from utils.callback import SMICallback

# Start a parser with the arguments
parser = argparse.ArgumentParser(description='Configuration for the training of the model')

# Parsing arguments if required in the shell pipeline
parser.add_argument('--baseline', action='store_true', help='argument for training the model with no or the most basic parameters')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
parser.add_argument('--model', type=str, default='resnet50', help='model to train; options: resnet50, convnextv1')
parser.add_argument('--seed', type=int, default=22, help='seed for the random number generator')

# Parse the arguments
args = parser.parse_args()


#####################################################################################
############################ Initialise Seed ########################################
#####################################################################################

if args.seed != 22:
    SEED = args.seed

#####################################################################################
############################ Initialise Parameters ##################################
#####################################################################################

parameters = random_config()

# Run training with the base line parameters
if args.baseline:
    parameters = base_line()

# Run training with random parameter configuration
if DEBUG:
    for key, value in parameters.items():
        print(key, ':', value)
    print('seed:', SEED)
    print('\n')

#####################################################################################
############################ Create Folders & Parameter File ########################
#####################################################################################

run_dir = create_subfolder()

# Store parameter configuration in yaml file
with open(os.path.join('runs', run_dir, 'parameters.yaml'), 'w') as f:
    # Model name
    f.write('model: '+str(args.model)+'\n')
    # Parameters
    for key, value in parameters.items():
        f.write(str(key)+': '+str(value)+'\n')
    # Seed
    f.write('seed : '+str(SEED)+'\n')

#####################################################################################
############################ Load data ##############################################
#####################################################################################

(x_train, y_train), (_, _) = tf.keras.datasets.cifar100.load_data()

#####################################################################################
############################ Preprocessing ##########################################
#####################################################################################

if DEBUG:
    x_train = x_train[:10000]
    y_train = y_train[:10000]
    print('Loaded:', x_train.shape, y_train.shape)
    print('Mean pixel value:',np.mean(x_train[0]))

# Robust scaling (median of 0 and std of 1)
if parameters['preprocessing'] == 'robust_scaling':
    x_train = x_train.astype('float32')
    q25, q75 = np.percentile(x_train, [25, 75], axis=(0, 1, 2))
    x_train = (x_train - np.median(x_train, axis=(0, 1, 2))) / (q75 - q25)
    if DEBUG:
        print('Mean pixel value of 1 sample after robust_scaling:',np.mean(x_train[0]), 'median:', np.median(x_train), 'std:', np.std(x_train))

# Standardization (mean of 0 and std of 1)
elif parameters['preprocessing'] == 'standardization':
    x_train = x_train.astype('float32')
    x_train = (x_train - np.mean(x_train, axis=(0, 1, 2))) / (np.std(x_train, axis=(0, 1, 2)))
    if DEBUG:
        print('Mean pixel value of 1 sample after standardization:',np.mean(x_train[0]), 'mean:', np.mean(x_train), 'std:', np.std(x_train))

# Minmax (press data between 0 and 1)
elif parameters['preprocessing'] == 'minmax':
    x_train = x_train.astype('float32')
    x_train = (x_train - np.min(x_train, axis=(0))) / (np.max(x_train, axis=(0)) - np.min(x_train, axis=(0)))
    if DEBUG:
        print('Mean pixel value of 1 sample after minmax:',np.mean(x_train[0]), 'mean:', np.mean(x_train), 'std:', np.std(x_train))

# No preprocessing
elif parameters['preprocessing'] == 'None':
    if DEBUG:
        print('No preprocessing applied')

y_train = tf.keras.utils.to_categorical(y_train, 100)

#####################################################################################
############################ Augmentation & Batching ################################
#####################################################################################

if DEBUG:
    print('Augmentation:', parameters['augmentation'])

# Cutmix function inspired by https://github.com/facebookresearch/ConvNeXt and https://www.kaggle.com/code/cdeotte/cutmix-and-mixup-on-gpu-tpu
if parameters['augmentation'] == 'cutmix':
    mix_ds1 = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024, seed=SEED + 2)
    mix_ds2 = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024, seed=SEED + 3)
    mix_ds = tf.data.Dataset.zip((mix_ds1, mix_ds2))
    data = mix_ds.map(lambda ds1, ds2: cutmix(ds1, ds2), num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(int(parameters['batch_size'])).prefetch(tf.data.experimental.AUTOTUNE)
    assert data.as_numpy_iterator().next()[0].shape[1:4] == x_train.shape[1:4]
    preprocessing_layer = Preprocessing(SEED, data.as_numpy_iterator().next()[0].shape, random=False)

    if DEBUG:
        image_batch, label_batch = next(iter(data))
        plt.figure(figsize=(5, 5))
        print('\n'+'Mix label map:')
        for i in range(9) if int(parameters['batch_size']) > 1 else range(1):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(tf.reshape(image_batch[i], (32, 32, 3)))
            loc = np.where(tf.reshape(label_batch[i], (100,)).numpy() > 0)[0]
            try: 
                first_loc = loc[0]
            except IndexError:
                first_loc = 0
            try:
                second_loc = loc[1]
            except IndexError:
                second_loc = 0
            try:
                first_label = float(np.round(tf.reshape(label_batch[i], (100,)).numpy()[loc[0]] * 100, 2))
            except IndexError:
                first_label = 0
            try:
                second_label = float(np.round(tf.reshape(label_batch[i], (100,)).numpy()[loc[1]] * 100, 2))
            except IndexError:
                second_label = 0
            if len(loc) == 1:
                second_label = 0
            print(f'Label {first_loc} = {first_label}% and Label {second_loc} = {second_label}%')
            plt.axis("off")
        plt.show()

    if DEBUG:
        print('\n')
        print('Mean pixel value of 1 sample after cutmix:',np.mean(data.as_numpy_iterator().next()[0][0]), 'mean of all:', np.mean(data.as_numpy_iterator().next()[0]), 'std of all:', np.std(data.as_numpy_iterator().next()[0]))

# Mixup function inspired by https://github.com/facebookresearch/ConvNeXt and https://www.kaggle.com/code/cdeotte/cutmix-and-mixup-on-gpu-tpu
elif parameters['augmentation'] == 'mixup':

    if parameters['preprocessing'] == 'None':
        x_train = x_train.astype('float32')
        x_train = x_train / 255.0

    if DEBUG:
        print('Mean pixel value before mixup:',np.mean(x_train[0]))

    mix_ds1 = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024, seed=SEED + 2)
    mix_ds2 = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024, seed=SEED + 3)
    mix_ds = tf.data.Dataset.zip((mix_ds1, mix_ds2))
    data = mix_ds.map(lambda ds1, ds2: mixup(ds1, ds2, 0.2), num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(int(parameters['batch_size'])).prefetch(tf.data.experimental.AUTOTUNE)
    # assert data.as_numpy_iterator().next()[0].shape == x_train.shape[1:4]
    preprocessing_layer = Preprocessing(SEED, data.as_numpy_iterator().next()[0].shape, random=False)

    if DEBUG:   
        image_batch, label_batch = next(iter(data))
        plt.figure(figsize=(5, 5))
        print('\n'+'Mix label map:')
        for i in range(9) if int(parameters['batch_size']) > 1 else range(1):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(tf.reshape(image_batch[i], (32, 32, 3)))
            loc = np.where(tf.reshape(label_batch[i], (100,)).numpy() > 0)[0]
            try: 
                first_loc = loc[0]
            except IndexError:
                first_loc = 0
            try:
                second_loc = loc[1]
            except IndexError:
                second_loc = 0
            try:
                first_label = float(np.round(tf.reshape(label_batch[i], (100,)).numpy()[loc[0]] * 100, 2))
            except IndexError:
                first_label = 0
            try:
                second_label = float(np.round(tf.reshape(label_batch[i], (100,)).numpy()[loc[1]] * 100, 2))
            except IndexError:
                second_label = 0
            if len(loc) == 1:
                second_label = 0
            print(f'Label {first_loc} = {first_label}% and Label {second_loc} = {second_label}%')
            plt.axis("off")
        plt.show()

    if DEBUG:
        print('\n')
        print('Mean pixel value of 1 sample after mixup:',np.mean(data.as_numpy_iterator().next()[0][0]), 'mean of all:', np.mean(data.as_numpy_iterator().next()[0]), 'std of all:', np.std(data.as_numpy_iterator().next()[0]))

    if parameters['preprocessing'] == 'None':
        # Convert back to 0-255
        data = data.map(lambda x, y: (x * 255, y), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if DEBUG:
            print('Mean pixel value of 1 sample after * 255:',np.mean(data.as_numpy_iterator().next()[0][0]), 'mean of all:', np.mean(data.as_numpy_iterator().next()[0]), 'std of all:', np.std(data.as_numpy_iterator().next()[0]))

# Random augmentation
elif parameters['augmentation'] == 'random':
    data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(int(parameters['batch_size'])).prefetch(tf.data.experimental.AUTOTUNE)
    preprocessing_layer = Preprocessing(SEED, data.as_numpy_iterator().next()[0].shape, random=True)

# No augmentation
elif parameters['augmentation'] == 'None': 
    data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(int(parameters['batch_size'])).prefetch(tf.data.experimental.AUTOTUNE)
    preprocessing_layer = Preprocessing(SEED, data.as_numpy_iterator().next()[0].shape, random=False)

# Shuffle the data
data = data.shuffle(1024, seed=SEED)

#####################################################################################
############################ Precision casting ######################################
#####################################################################################

# Global policy float16 precision casting
if parameters['precision'] == 'global_policy_float16':
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Cast the data to the precision specified in the argument precision (datatypy is the same as the parameter value)
else:
    data = data.map(lambda x, y: (tf.cast(x, tf.dtypes.as_dtype(str(parameters['precision']))), y))

#####################################################################################
############################ Partitioning ###########################################
#####################################################################################

split = [round(float(x) * 0.01, 2) for x in parameters['partitioning'].split('-')]

# Split the data according to the parameter value
train_size = int(split[0] * len(data))
train_ds = data.take(train_size)
val_ds = data.skip(train_size).take(int(split[1] * len(data)))
test_ds = data.skip(int(round(split[0] + split[1],2) * len(data))).take(int(split[1] * len(data)))

# Logging the size of the data
if parameters['batch_size'] == '1':
    print(f'Size of the training set: {len(train_ds)} (unbatched)')
    print(f'Size of the validation set: {len(val_ds)} (unbatched)')
    print(f'Size of the test set: {len(test_ds)} (unbatched)')
else:
    print(f'Size of the training set: {len(train_ds)} (batched), {len(train_ds) * int(parameters["batch_size"])} (unbatched)')
    print(f'Size of the validation set: {len(val_ds)} (batched), {len(val_ds) * int(parameters["batch_size"])} (unbatched)')
    print(f'Size of the test set: ', len(test_ds), f' (batched), {len(test_ds) * int(parameters["batch_size"])} (unbatched)')

#####################################################################################
############################ Model building #########################################
#####################################################################################

# Loading resnet50 model
if args.model == 'resnet50':
    model = load_resnet50(classes=100, input_shape=data.as_numpy_iterator().next()[0].shape[1:4], weights=None)

# Loading version 1 of the convnext model
elif args.model == 'convnextv1':
    model = load_convnextv1(classes=100, input_shape=data.as_numpy_iterator().next()[0].shape[1:4], weights=None)

# Initialize model, stack the preprocessing layer and the model
combined_model = tf.keras.Sequential([
    preprocessing_layer,
    model
])

#####################################################################################
############################ Learning Rate ##########################################
#####################################################################################

# Set learning rate according to the parameter value
learning_rate = float(parameters['lr'])

#####################################################################################
############################ Learning Rate Schedule #################################
#####################################################################################

# Constant learning rate
if parameters['lr_schedule'] == 'constant':
    learning_rate_schedule = learning_rate

# Exponential learning rate
elif parameters['lr_schedule'] == 'exponential':
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate, 
        decay_steps=1000, # This should be another parameter
        decay_rate=0.96,
        staircase=True)
    
# Polynomial learning rate
elif parameters['lr_schedule'] == 'polynomial':
    learning_rate_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        learning_rate, 
        decay_steps=1000,)

# Cosine learning rate
elif parameters['lr_schedule'] == 'cosine':
    learning_rate_schedule = tf.keras.optimizers.schedules.CosineDecay(
        learning_rate, 
        decay_steps=1000,)

#####################################################################################
############################ Optimizer Momentum #####################################
#####################################################################################

# Set optimizer momentum according to the parameter value
optimizer_momentum = parameters['optimizer_momentum']

#####################################################################################
############################ Optimizer ##############################################
#####################################################################################

# Set SGD optimizer
if parameters['optimizer'] == 'SGD':
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate_schedule,
        momentum=float(optimizer_momentum))

# Set RMSProp optimizer
elif parameters['optimizer'] == 'RMSProp':
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=learning_rate_schedule,
        momentum=float(optimizer_momentum))
    
# Set Adam optimizer
elif parameters['optimizer'] == 'Adam':
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_schedule)
    
# Set AdamW optimizer with standard weight decay (from tf docs)
elif parameters['optimizer'] == 'AdamW':
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate_schedule, 
        weight_decay=0.001)

#####################################################################################
############################ Pre-quantization #######################################
#####################################################################################

# Pre-quantize the model if specified in the parameter value
if parameters['internal'] == 'pre_quantization' and parameters['precision'] != 'global_policy_float16':

    # Convert the data to float16 (needed for quantization)
    train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.dtypes.as_dtype('float16')), y))
    val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.dtypes.as_dtype('float16')), y))
    test_ds = test_ds.map(lambda x, y: (tf.cast(x, tf.dtypes.as_dtype('float16')), y))

    quantize_model = tfmot.quantization.keras.quantize_model
    combined_model = quantize_model(combined_model.layers[1])

    print('\n'+'Pre-quantizing model.'+'\n')

elif parameters['internal'] == 'pre_quantization' and parameters['precision'] == 'global_policy_float16':
    print('\n'+'Not quantizing because of global policy float16.'+'\n')

else: 
    print('\n'+'No pre-quantizing.'+'\n')

#####################################################################################
############################ Training ###############################################
#####################################################################################

# Build the model
combined_model.build(input_shape=(None, 
                    data.as_numpy_iterator().next()[0].shape[1],
                    data.as_numpy_iterator().next()[0].shape[2],
                    data.as_numpy_iterator().next()[0].shape[3])) 

# Compile the model
if parameters['internal'] == 'jit_compilation':
    combined_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'],
        # Set jit_compile to True if specified in the parameter value
        jit_compile=True
    )
else:
    combined_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'],
        # Set jit_compile to False if other parameter values are switched on
        jit_compile=False
    )

# Train the model
combined_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
    # Call the SMICallback callback function for logging GPU and training metrics
    callbacks=[SMICallback()],
)

#####################################################################################
############################ Post Quantization ######################################
#####################################################################################

if parameters['internal'] == 'post_quantization' and parameters['precision'] != 'global_policy_float16':

    # Convert the data to float16 (needed for quantization)
    train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.dtypes.as_dtype('float16')), y))
    val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.dtypes.as_dtype('float16')), y))
    test_ds = test_ds.map(lambda x, y: (tf.cast(x, tf.dtypes.as_dtype('float16')), y))

    quantize_model = tfmot.quantization.keras.quantize_model
    combined_model = quantize_model(combined_model.layers[1])

    combined_model.compile(
            optimizer = optimizer, 
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy'],
            jit_compile=False
    )

    print('\n'+'Post-quantized model.'+'\n')

elif parameters['internal'] == 'post_quantization' and parameters['precision'] == 'global_policy_float16':
    print('\n'+'Not quantizing because of global policy float16.'+'\n')

#####################################################################################
############################ Testing ################################################
#####################################################################################

# Evaluate the model on the test data using `evaluate`
if DEBUG:
    print('Evaluating model.')
    test_loss, test_acc = combined_model.evaluate(test_ds, verbose=2)
    print('\nTest accuracy:', test_acc)
else:
    test_loss, test_acc = combined_model.evaluate(test_ds, verbose=0)

# Save the model and #parameters (model size)
with open(os.path.join('runs', run_dir, 'parameters.yaml'), 'a') as f:
    f.write('n_parameters: ' + str(combined_model.count_params())+'\n')

# Save the model test set
with open(os.path.join('runs', run_dir, 'parameters.yaml'), 'a') as f:
    f.write('test_accuracy: ' + str(round(test_acc, 3)) + '\n')

#####################################################################################