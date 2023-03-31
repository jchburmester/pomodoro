""" Implementation of the ConvNext model. 

Inspired by: https://github.com/keras-team/keras/blob/v2.12.0/keras/applications/convnext.py#L562

"""

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_convnext(classes, input_shape):
    model = convnext_small(num_classes=classes, cardinality=1, input_shape=input_shape)
    return model

# ---------------------------- ConvNext ----------------------------

class ConvNeXtBottleneck(tf.keras.layers.Layer):
    def __init__(self, filters, strides, cardinality=1):
        super(ConvNeXtBottleneck, self).__init__()
        self.filters = filters
        self.strides = strides
        self.cardinality = cardinality

        # 1x1 convolution layer
        self.conv1 = tf.keras.layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.Activation('relu')

        # 3x3 convolution layer with cardinality
        self.conv2 = tf.keras.layers.Conv2D(filters, (3, 3), strides=strides, padding='same', groups=cardinality)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.Activation('relu')

        # 1x1 convolution layer
        self.conv3 = tf.keras.layers.Conv2D(2 * filters, (1, 1), strides=(1, 1), padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()

        # Shortcut connection for residual learning
        self.shortcut = self._build_shortcut()
        self.add = tf.keras.layers.Add()
        self.relu3 = tf.keras.layers.Activation('relu')

    def _build_shortcut(self):
        input_filters = 64 if self.filters == 128 else self.filters // 2
        if self.strides == (1, 1) and input_filters == self.filters:
            return lambda x: x
        else:
            return tf.keras.layers.Conv2D(2 * self.filters, (1, 1), strides=self.strides, padding='same')

    def call(self, inputs, **kwargs):
        # First stage of the bottleneck block
        x = self.relu1(self.bn1(self.conv1(inputs)))
        # Second stage of the bottleneck block
        x = self.relu2(self.bn2(self.conv2(x)))
        # Third stage of the bottleneck block
        x = self.bn3(self.conv3(x))

        # Shortcut connection
        shortcut = self.shortcut(inputs)

        # Add the shortcut connection to the output and apply ReLU activation
        return self.relu3(self.add([x, shortcut]))


class ConvNeXt(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes=10, cardinality=1, input_shape=(32, 32, 3)):
        super(ConvNeXt, self).__init__()
        self.cardinality = cardinality
        
        # Initial 3x3 convolution
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.Activation('relu')

        # Build the three layers with varying number of bottleneck blocks
        self.layer2 = self._build_layer(block, 64, num_blocks[0], stride=1)
        self.layer3 = self._build_layer(block, 128, num_blocks[1], stride=2)
        self.layer4 = self._build_layer(block, 256, num_blocks[2], stride=2)

        # Global average pooling layer
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        # Fully connected layer for classification
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def _build_layer(self, block, filters, num_blocks, stride):
        # Build a layer consisting of multiple bottleneck blocks with given strides
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(filters=filters, strides=(stride, stride), cardinality=self.cardinality))
        return tf.keras.Sequential(layers)

    def call(self, inputs, **kwargs):
        # Initial convolution and activation
        x = self.relu1(self.bn1(self.conv1(inputs)))
        # First layer of bottleneck blocks
        x = self.layer2(x)
        # Second layer of bottleneck blocks
        x = self.layer3(x)
        # Third layer of bottleneck blocks
        x = self.layer4(x)

        # Global average pooling
        x = self.avg_pool(x)
        # Fully connected layer for classification
        x = self.fc(x)

        return x

# Function to create a small ConvNeXt model
def convnext_small(num_classes=10, cardinality=1, input_shape=(32, 32, 3)):
    return ConvNeXt(ConvNeXtBottleneck, num_blocks=[2, 2, 2], num_classes=num_classes, cardinality=cardinality, input_shape=input_shape)

if __name__ == '__main__':
    # Usage example
    model = convnext_small(num_classes=10, cardinality=1, input_shape=(32, 32, 3))
    input_shape = (None, 32, 32, 3)  # Example input shape for a 32x32 RGB image
    model.build(input_shape)
    model.summary()