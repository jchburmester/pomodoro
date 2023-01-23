""" https://keras.io/examples/vision/mixup/ """
import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@tf.function
def mixup(ds1, ds2, alpha):

    images_one, labels_one = ds1
    images_two, labels_two = ds2
    batch_size = tf.shape(images_one)[0]

    def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
        gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
        gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
        return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

    # Sample lambda and reshape it to do the mixup
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    images = images_one * x_l + images_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)
    return (images, labels)
    
if __name__ == '__main__':
    # Test the mixup function with images from cifar10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_train = tf.reshape(x_train, (-1, 32, 32, 3))
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    mix_ds1 = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024).batch(9)
    mix_ds2 = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024).batch(9)
    mix_ds = tf.data.Dataset.zip((mix_ds1, mix_ds2))

    train_mu = mix_ds.map(lambda ds1, ds2: mixup(ds1, ds2, 0.2), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    sample_images, sample_labels = next(iter(train_mu))
    plt.figure(figsize=(10, 10))
    for i, (image, label) in enumerate(zip(sample_images[:9], sample_labels[:9])):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image.numpy().squeeze())
        print(label.numpy().tolist())
        plt.axis("off")
    plt.show()