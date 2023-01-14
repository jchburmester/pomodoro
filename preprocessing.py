""" 
Main file for preprocessing
Pomodoro, 4.12.2022
"""
import tensorflow as tf
import tensorflow.keras.layers as layer
from random import sample
from random import seed

class Preprocessing(tf.keras.layers.Layer):

    def __init__(self, seed, data_shape=(256,256,3), config=None):
        """
        Constructor.

        Parameters:
        ----------
            data_shape : tuple
                shape of input data. Defaults to standard RGB image shape.
            normalize : bool
                if normalization should be applied.
            scale : bool
                if scaling should be applied.
            augment : bool
                if augmentation should be applied.
        """
        super(Preprocessing, self).__init__()
        
        self.seed_val = seed

        # collects preprocessing steps
        self.prepro_layers = []
        
        # data normalization
        if config == 'normalization':
            self.prepro_layers.append(layer.Normalization())

        # data augmentation
        self.possible_augmentation_steps = {
            "Random Flipping" : layer.RandomFlip(seed=self.seed_val), 
            "Random Rotation" : layer.RandomRotation(0.3, seed=self.seed_val), 
            "Random Zoom" : layer.RandomZoom(0.2, seed=self.seed_val),
            "Random Contrast" : layer.RandomContrast(0.3, seed=self.seed_val), 
            "Random Translation" : layer.RandomTranslation(0.2,0.2, seed=self.seed_val), 
            "Random Height Shift" : layer.RandomHeight(0.2, seed=self.seed_val), 
            "Random Width Shift" : layer.RandomWidth(0.2, seed=self.seed_val)}

        # if augmentation should be applied, randomly select 3 augmentation methods
        if config == 'augmentation':
            self.augmentation_subset = sample(list(self.possible_augmentation_steps.items()), 3)

            # to collect names of applied methods
            self.aug_names = []

            # first part of each item is name, second part method
            for i in self.augmentation_subset:
                self.aug_names.append(i[0])
                self.prepro_layers.append(i[1])
            
            print("Applied augmentation steps: ")
            print(*self.aug_names, sep=", ")

        # placeholder layer if no preprocessing layer is added
        if len(self.prepro_layers) == 0:
            self.prepro_layers.append(layer.Reshape(data_shape))
            print("No preprocessing steps applied.")

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