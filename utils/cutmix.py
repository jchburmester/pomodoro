""" https://keras.io/examples/vision/cutmix/ """
import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@tf.function
def cutmix(ds1, ds2):

    images_one, labels_one = ds1
    images_two, labels_two = ds2
    batch_size = tf.shape(images_one)[0]
    img_size = tf.shape(images_one)[1]
    img_size = tf.cast(img_size, tf.int32)

    alpha = [0.25]
    beta = [0.25]

    def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
        gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
        gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
        return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

    def get_box(lambda_value):
        cut_rat = tf.math.sqrt(1.0 - lambda_value)
        cut_rat = tf.cast(cut_rat, tf.int32)

        cut_w = img_size * cut_rat  # rw
        cut_w = tf.cast(cut_w, tf.int32)

        cut_h = img_size * cut_rat  # rh
        cut_h = tf.cast(cut_h, tf.int32)

        cut_x = tf.random.uniform((1,), minval=0, maxval=img_size, dtype=tf.int32)  # rx
        cut_y = tf.random.uniform((1,), minval=0, maxval=img_size, dtype=tf.int32)  # ry

        boundaryx1 = tf.clip_by_value(cut_x[0] - cut_w // 2, 0, img_size)
        boundaryy1 = tf.clip_by_value(cut_y[0] - cut_h // 2, 0, img_size)
        bbx2 = tf.clip_by_value(cut_x[0] + cut_w // 2, 0, img_size)
        bby2 = tf.clip_by_value(cut_y[0] + cut_h // 2, 0, img_size)

        target_h = bby2 - boundaryy1
        if target_h == 0:
            target_h += 1

        target_w = bbx2 - boundaryx1
        if target_w == 0:
            target_w += 1

        return boundaryx1, boundaryy1, target_h, target_w

    lambda_value = sample_beta_distribution(1, alpha, beta)

    # Define Lambda
    lambda_value = lambda_value[0][0]

    # Get the bounding box offsets, heights and widths
    boundaryx1, boundaryy1, target_h, target_w = get_box(lambda_value)

    # Get a patch from the second image (`images_two`)
    crop2 = tf.image.crop_to_bounding_box(images_two, boundaryy1, boundaryx1, target_h, target_w)

    # Pad the `images_two` patch (`crop2`) with the same offset
    images_two = tf.image.pad_to_bounding_box(crop2, boundaryy1, boundaryx1, img_size, img_size)

    # Get a patch from the first image (`images_one`)
    crop1 = tf.image.crop_to_bounding_box(images_one, boundaryy1, boundaryx1, target_h, target_w)

    # Pad the `images_one` patch (`crop1`) with the same offset
    img1 = tf.image.pad_to_bounding_box(crop1, boundaryy1, boundaryx1, img_size, img_size)

    # Modify the first image by subtracting the patch from `images_one`
    # (before applying the `images_two` patch)
    images_one = images_one - img1
    # Add the modified `images_one` and `images_two`  together to get the CutMix image
    image = images_one + images_two

    # Adjust Lambda in accordance to the pixel ration
    lambda_value = 1 - (target_w * target_h) / (img_size * img_size)
    lambda_value = tf.cast(lambda_value, tf.float32)

    # Combine the labels of both images
    label = lambda_value * labels_one + (1 - lambda_value) * labels_two

    return image, label

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    mix_ds1 = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024).batch(32)
    mix_ds2 = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024).batch(32)
    mix_ds = tf.data.Dataset.zip((mix_ds1, mix_ds2))

    train_cm = mix_ds.map(lambda ds1, ds2: cutmix(ds1, ds2), num_parallel_calls=tf.data.AUTOTUNE)

    # Let's preview 9 samples from the dataset
    image_batch, label_batch = next(iter(train_cm))
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(tf.reshape(image_batch[i], (32, 32, 3)))
        plt.axis("off")
    plt.show()