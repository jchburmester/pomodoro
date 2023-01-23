import tensorflow as tf

# Import the ResNet50 model
def load_model(classes, input_shape, weights=None):
    model = tf.keras.applications.resnet50.ResNet50(classes=classes, input_shape=input_shape, weights=weights)
    return model