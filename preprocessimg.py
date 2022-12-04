import tensorflow as tf
import tensorflow.keras.layers as layer
from random import sample
from random import seed

class Preprocessing(tf.keras.layers.Layer):

    def __init__(self, data_shape=(256,256,3), normalize=False, scale=False, augment=False):
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
        
        self.seed_val = seed(22)

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
            if(data_shape[-1]==3 and len(data_shape)==3):
                self.prepro_layers.append(layer.Rescaling(scale=1./255))
            else:
                print("Make sure that input images are in RGB format.")

        # data augmentation
        self.possible_augmentation_steps = {
            "Random Flipping" : layer.RandomFlip(seed=self.seed_val), 
            "Random Rotation" : layer.RandomRotation(0.3, seed=self.seed_val), 
            "Random Zoom" : layer.RandomZoom(0.2, seed=self.seed_val),
            "Random Contrast" : layer.RandomContrast(0.3, seed=self.seed_val), 
            "Random Translation" : layer.RandomTranslation(0.2,0.2, seed=self.seed_val), 
            "Radom Height Shift" : layer.RandomHeight(0.2, seed=self.seed_val), 
            "Random Width Shift" : layer.RandomWidth(0.2, seed=self.seed_val)}

        # if augmentation should be applied, randomly select 3 augmentation methods
        if(augment):
            self.aug_list = list(self.possible_augmentation_steps.items())
            self.seed_val
            self.augmentation_subset = sample(self.aug_list, 3)

            # to collect names of applied methods
            self.aug_names = []

            # first part of each item is name, second part method
            for i in self.augmentation_subset:
                self.aug_names.append(i[0])
                self.prepro_layers.append(i[1])
            
            print("Applied augmentation steps: ")
            print(*self.aug_names, sep=", ")

        # placeholder layer if no preprocessing layer is added
        if(len(self.prepro_layers)==0):
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