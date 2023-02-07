import tensorflow as tf

# Import the ConvNext model
def load_convnextv1(classes, input_shape, weights=None):
    model = tf.keras.applications.convnext.ConvNeXtBase(classes=classes, input_shape=input_shape, weights=weights)
    return model