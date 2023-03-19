""" Implementation of the ResNet50 model. """

"""
CHANGED (from last years implementation):
- input_shape now adjustable parameter (last year: fixed to (224,224,3))
- Batch Normalization Epsilon from default (0.001) --> 1.001e-5 (see tf.keras.ResNet50)
- More modularity (last year: one big class, now: blocks and ResNet50 class)

(- other difference: tf.keras.ResNet50 is initialized with ImageNet weights and 1000 classes per default, but we called with no pre-training so difference negligible)
"""

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_resnet50(classes, input_shape):
    model = ResNet50(num_classes=classes, input_shape=input_shape)
    return model


# ---------------------------- ResNet50 ----------------------------

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

class Block(tf.keras.Model):
    def __init__(self, input_channels, output_channels, identity_block=False, identity_strides=1):
        super(Block, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(input_channels, (1, 1), strides=identity_strides)
        self.conv2 = tf.keras.layers.Conv2D(input_channels, (3, 3), padding='same')
        self.conv3 = tf.keras.layers.Conv2D(output_channels, (1, 1), padding='same')

        if identity_block is False:
            self.shortcut = tf.keras.layers.Conv2D(output_channels, (1, 1), strides=identity_strides, padding='same')

        self.bn1 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        self.bn2 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        self.bn3 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        self.bn4 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)

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
            
        y = tf.keras.layers.Add()([input, x])
        y = tf.nn.relu(y)

        return y

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

class ResNet50(tf.keras.Model):
    def __init__(self, num_classes, input_shape):
        super(ResNet50, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(input_shape=input_shape, filters=64, kernel_size=(7, 7), strides=2, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        self.maxpool1 = tf.keras.layers.MaxPool2D((3, 3), strides=2, padding='same')

        self.blocks = []
        # First set of blocks
        self.blocks.append(Block(64, 256, False, 1))
        self.blocks.extend([Block(64, 256, True, 1) for _ in range(2)])

        # Second set of blocks
        self.blocks.append(Block(128, 512, False, 2))
        self.blocks.extend([Block(128, 512, True, 1) for _ in range(3)])

        # Third set of blocks
        self.blocks.append(Block(256, 1024, False, 2))
        self.blocks.extend([Block(256, 1024, True, 1) for _ in range(5)])

        # Fourth set of blocks
        self.blocks.append(Block(512, 2048, False, 2))
        self.blocks.extend([Block(512, 2048, True, 1) for _ in range(2)])

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.maxpool1(x)

        for block in self.blocks:
            x = block(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    model = ResNet50(100, (224, 224, 3))
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()

    x = tf.keras.Input(shape=(224, 224, 3), dtype=tf.float32)
    print('input', x.shape)
    for layer in model.layers:
        x = layer(x)
        print(layer.name, x.shape)