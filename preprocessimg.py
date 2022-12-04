import tensorflow as tf
import tensorflow.keras.layers as layer
from random import sample
from random import seed

class Preprocessing(tf.keras.Layer):

    def __init__(self, normalize=False, scale=False, augment=False):
        """
        Constructor.

        Parameters:
        ----------
            normalize : bool
                if normalization should be applied.
            scale : bool
                if scaling should be applied.
            augment : bool
                if augmentation should be applied.
        """
        super(Preprocessing, self).__init__()

        # collects preprocessing steps
        self.prepro_layers = []

        # make sure that normalization and scaling are on their own
        if(scale and normalize):
            print("Normalization and Scaling cannot be applied at the same time.")
        
        # data normalization
        elif(normalize):
            self.prepro_layers.append(layer.Normalization())

        # data scaling. Scales to range [0,1]
        elif(scale):
            self.prepro_layers.append(layer.Rescaling(scale=1./255))

        # data augmentation
        # if augmentation should be applied, randomly select 3 augmentation methods
        self.possible_augmentation_steps = [layer.RandomFlip(seed=10), layer.RandomRotation(0.3, seed=9), layer.RandomZoom(0.2, seed=8),
                                layer.RandomContrast(0.3, seed=7), layer.RandomTranslation(0.2,0.2, seed=6), layer.RandomHeight(0.2, seed=5), 
                                layer.RandomWidth(0.2, seed=4)]

        if(augment):
            seed(22)
            self.augmentation_subset = sample(self.augmentation_steps, 3)

            for i in self.augmentation_subset:
                self.prepro_layers.append(i)

        # placeholder layer if no preprocessing layer is added
        if(len(self.prepro_layers)==0):
            self.prepro_layers.append(layer.Reshape((256,256,3)))



    @tf.function
    def call(self,x):
        """
        Applies preprocessing steps to data.

        Parameters:
        ----------
            x : tensor
                data.

        Returns:
        --------
            x : tensor
                preprocessed data.
        """
        for layer in self.prepro_layers:
            x = layer(x)

        return x