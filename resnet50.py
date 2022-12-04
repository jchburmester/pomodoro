""" 
The ResNet50 model class 
Pomodoro, 2.12.2022
"""

import tensorflow as tf
import tensorflow.keras.layers as layers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ResNet50

# Base layer
# ConvBlock
# IdentityBlock x2
# ConvBlock
# IdentityBlock x3
# ConvBlock
# IdentityBlock x5
# ConvBlock
# IdentityBlock x2
# AvgPool
# FC

# Blocks for the ResNet50

# [1x1, 64]
# [3x3, 64]  x 3 
# [1x1, 256]

# [1x1, 128]
# [3x3, 128] x 4
# [1x1, 512]

# [1x1, 256]
# [3x3, 256] x 6
# [1x1, 1024]

# [1x1, 512]
# [3x3, 512] x 3
# [1x1, 2048]

class Block(layers.Layer):
    def __init__(self, input_channels, output_channels, type, identity_strides=1):
        super(Block, self).__init__()
        self.conv1 = layers.Conv2D(input_channels, (1, 1), strides=identity_strides)
        self.conv2 = layers.Conv2D(input_channels, (3, 3), padding='same')
        self.conv3 = layers.Conv2D(output_channels, (1, 1), padding='same')
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.bn4 = layers.BatchNormalization()

        if type == 'convolutional':
            self.shortcut = layers.Conv2D(output_channels, (1, 1), strides=identity_strides, padding='same')

    @tf.function()
    def call(self, input):

        x = self.conv1(input)
        x = self.bn1(x)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        # identity shortcut
        if hasattr(self, 'shortcut'):
            input = self.shortcut(input)
            input = self.bn4(input)
            
        y = layers.Add()([x, input])
        y = tf.nn.relu(y)

        return y


class ResNet50(tf.keras.Model):
    def __init__(self, num_classes, input_shape, jit_compilation=False):
        super(ResNet50, self).__init__()
        self.conv1 = layers.Conv2D(input_shape=input_shape, filters=64, kernel_size=(7, 7), strides=2, padding='same')
        self.bn = layers.BatchNormalization()
        self.maxpool1 = layers.MaxPool2D((3, 3), strides=2, padding='same')

        self.block1_1 = Block(64, 256, 'convolutional', 1)
        self.block1_2 = Block(64, 256, 'identity', 1)
        self.block1_3 = Block(64, 256, 'identity', 1)

        self.block2_1 = Block(128, 512, 'convolutional', 2)
        self.block2_2 = Block(128, 512, 'identity', 1)
        self.block2_3 = Block(128, 512, 'identity', 1)
        self.block2_4 = Block(128, 512, 'identity', 1)

        self.block3_1 = Block(256, 1024, 'convolutional', 2)
        self.block3_2 = Block(256, 1024, 'identity', 1)
        self.block3_3 = Block(256, 1024, 'identity', 1)
        self.block3_4 = Block(256, 1024, 'identity', 1)
        self.block3_5 = Block(256, 1024, 'identity', 1)
        self.block3_6 = Block(256, 1024, 'identity', 1)

        self.block4_1 = Block(512, 2048, 'convolutional', 2)
        self.block4_2 = Block(512, 2048, 'identity', 1)
        self.block4_3 = Block(512, 2048, 'identity', 1)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes, activation='softmax')

    @tf.function()
    def call(self, input):
        x = self.conv1(input)
        x = self.bn(x)
        x = tf.nn.relu(x)
        x = self.maxpool1(x)

        x = self.block1_1(x)
        x = self.block1_2(x)
        x = self.block1_3(x)

        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.block2_3(x)
        x = self.block2_4(x)

        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        x = self.block3_5(x)
        x = self.block3_6(x)

        x = self.block4_1(x)
        x = self.block4_2(x)
        x = self.block4_3(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x