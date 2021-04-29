import os
import time
import logging

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from keras import Input, Model
from keras.applications import InceptionResNetV2
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Reshape, concatenate, LeakyReLU, Lambda, \
    Activation, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_preprocessing import image
from scipy.io import loadmat

def build_encoder(input_shape=(64, 64, 3)):
    """
    Encoder Network
    """
    input_layer = Input(shape=input_shape)

    # 1st Convolutional Block
    enc = Conv2D(filters=32, kernel_size=5, strides=2, padding='same')(input_layer)
    # enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # 2nd Convolutional Block
    enc = Conv2D(filters=64, kernel_size=5, strides=2, padding='same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # 3rd Convolutional Block
    enc = Conv2D(filters=128, kernel_size=5, strides=2, padding='same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # 4th Convolutional Block
    enc = Conv2D(filters=256, kernel_size=5, strides=2, padding='same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # Flatten layer
    enc = Flatten()(enc)

    # 1st Fully Connected Layer
    enc = Dense(4096)(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # Second Fully Connected Layer
    enc = Dense(100)(enc)

    # Create a model
    model = Model(inputs=[input_layer], outputs=[enc])
    return model


def build_generator():
    """
    Create a Generator Model with hyperparameters values defined as follows
    """
    latent_dims = 100
    num_classes = 6

    input_z_noise = Input(shape=(latent_dims,))
    input_label = Input(shape=(num_classes,))

    x = concatenate([input_z_noise, input_label])

    x = Dense(2048, input_dim=latent_dims + num_classes)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)

    x = Dense(256 * 8 * 8)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)

    x = Reshape((8, 8, 256))(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=5, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=5, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=5, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=3, kernel_size=5, padding='same')(x)
    x = Activation('tanh')(x)

    model = Model(inputs=[input_z_noise, input_label], outputs=[x])
    return model


def expand_label_input(x):
    x = K.expand_dims(x, axis=1)
    x = K.expand_dims(x, axis=1)
    x = K.tile(x, [1, 64, 64, 1])
    return x


def build_discriminator(input_shape=(64, 64, 3), label_shape=(6,)):
    """
    Create a Discriminator Model with hyperparameters values defined as follows
    """
    image_input = Input(shape=input_shape)
    label_input = Input(shape=label_shape)

    x = Conv2D(64, kernel_size=3, strides=2, padding='same')(image_input)
    x = LeakyReLU(alpha=0.2)(x)

    label_input1 = Lambda(expand_label_input)(label_input)
    x = concatenate([x, label_input1], axis=3)

    x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(512, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[image_input, label_input], outputs=[x])
    return model


def build_fr_combined_network(encoder, generator, fr_model, input_shape=(64, 64, 3), label_shape=(6,)):
    input_image = Input(shape=input_shape)
    input_label = Input(shape=label_shape)

    latent0 = encoder(input_image)

    gen_images = generator([latent0, input_label])

    fr_model.trainable = False

    resized_images = Lambda(lambda x: K.resize_images(gen_images, height_factor=2, width_factor=2,
                                                      data_format='channels_last'))(gen_images)
    embeddings = fr_model(resized_images)

    model = Model(inputs=[input_image, input_label], outputs=[embeddings])
    return model


def build_fr_model(input_shape):
    resent_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
    image_input = resent_model.input
    x = resent_model.layers[-1].output
    out = Dense(128)(x)
    embedder_model = Model(inputs=[image_input], outputs=[out])

    input_layer = Input(shape=input_shape)

    x = embedder_model(input_layer)
    output = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x)

    model = Model(inputs=[input_layer], outputs=[output])
    return model


def build_image_resizer(input_shape=(64, 64, 3)):
    input_layer = Input(shape=input_shape)

    resized_images = Lambda(lambda x: K.resize_images(x, height_factor=3, width_factor=3,
                                                      data_format='channels_last'))(input_layer)

    model = Model(inputs=[input_layer], outputs=[resized_images])
    return model


def load_images(data_dir, image_paths, image_shape):
    logging.info("Starting to load images...")
    images = None

    for i, image_path in enumerate(image_paths):
        if i % 1000 == 0:
            logging.debug(f"Finish to load {i} images...")
        try:
            # Load image
            loaded_image = image.load_img(os.path.join(data_dir, image_path), target_size=image_shape)

            # Convert PIL image to numpy ndarray
            loaded_image = image.img_to_array(loaded_image)

            # Add another dimension (Add batch dimension)
            loaded_image = np.expand_dims(loaded_image, axis=0)

            # Concatenate all images into one tensor
            if images is None:
                images = loaded_image
            else:
                images = np.concatenate([images, loaded_image], axis=0)
        except Exception as e:
            logging.error("Error:", i, e)

    logging.info("Finish to load images...")
    return images

def load_data(source_file_path):
    """ Read the source_file, and return the list of filenames and a list of images (int)
    """
    with open(source_file_path) as f:
        images, age_list = [], []
        for line in f.readlines():
            image, age = line.split()
            images.append(image)
            age_list.append(int(age))
    return images, age_list


def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def write_log(callback, name, value, batch_no):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()


def save_rgb_img(img, path):
    """
    Save an rgb image
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Image")

    plt.savefig(path)
    plt.close()


def main():
    # Define hyperparameters
    # data_dir = "/content/data/UTKFaces2"
    data_dir = "./data/processed/UTKFaces2/"
    # source_file_path = "/content/data/source_file.txt"
    source_file_path = "./data/source_file.txt"
    # results_path = "/content/drive/MyDrive/face_aging/age_cgan/results"
    results_path = "./results"
    # saved_models_path = "/content/drive/MyDrive/face_aging/age_cgan/saved_models"
    saved_models_path = "./saved_models"

    images_path = "./data/images.npy"

    if not os.path.exists(results_path):
      os.makedirs(results_path)
    if not os.path.exists(saved_models_path):
      os.makedirs(saved_models_path)

    # wiki_dir = os.path.join(data_dir, "wiki_crop1")
    epochs = 500
    batch_size = 2
    image_shape = (128, 128, 3)
    z_shape = 100
    TRAIN_GAN = True
    TRAIN_ENCODER = False
    TRAIN_GAN_WITH_FR = False
    fr_image_shape = (192, 192, 3)

    # Define optimizers
    dis_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)
    gen_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)
    adversarial_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)

    """
    Build and compile networks
    """
    # Build and compile the discriminator network
    discriminator = build_discriminator(input_shape=image_shape)
    discriminator.compile(loss=['binary_crossentropy'], optimizer=dis_optimizer)

    # Build and compile the generator network
    generator = build_generator()
    generator.compile(loss=['binary_crossentropy'], optimizer=gen_optimizer)

    # Build and compile the adversarial model
    discriminator.trainable = False
    input_z_noise = Input(shape=(100,))
    input_label = Input(shape=(6,))
    recons_images = generator([input_z_noise, input_label])
    valid = discriminator([recons_images, input_label])
    adversarial_model = Model(inputs=[input_z_noise, input_label], outputs=[valid])
    adversarial_model.compile(loss=['binary_crossentropy'], optimizer=gen_optimizer)

    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
    tensorboard.set_model(generator)
    tensorboard.set_model(discriminator)

    """
    Load the dataset
    """
    # images, age_list = load_data(data_dir=data_dir, dataset="wiki")
    
    images, age_list = load_data(source_file_path)

    # age_cat = age_to_category(age_list)
    age_cat = age_list

    final_age_cat = np.reshape(np.array(age_cat), [len(age_cat), 1])
    classes = len(set(age_cat))
    y = to_categorical(final_age_cat, num_classes=len(set(age_cat)))

    try:
        loaded_images = np.load(images_path, allow_pickle=True)
    except:
        loaded_images = load_images(data_dir, images, (image_shape[0], image_shape[1]))
        np.save(images_path, loaded_images, allow_pickle=True)

    # Implement label smoothing
    real_labels = np.ones((batch_size, 1), dtype=np.float32) * 0.9
    fake_labels = np.zeros((batch_size, 1), dtype=np.float32) * 0.1

    """
    Train the generator and the discriminator network
    """
    if TRAIN_GAN:
        for epoch in range(epochs):
            print("Epoch:{}".format(epoch))

            gen_losses = []
            dis_losses = []

            number_of_batches = int(len(loaded_images) / batch_size)
            print("Number of batches:", number_of_batches)
            for index in range(number_of_batches):
                print("Batch:{}".format(index + 1))

                images_batch = loaded_images[index * batch_size:(index + 1) * batch_size]
                images_batch = images_batch / 127.5 - 1.0
                images_batch = images_batch.astype(np.float32)

                y_batch = y[index * batch_size:(index + 1) * batch_size]
                z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))

                """
                Train the discriminator network
                """

                # Generate fake images
                initial_recon_images = generator.predict_on_batch([z_noise, y_batch])

                d_loss_real = discriminator.train_on_batch([images_batch, y_batch], real_labels)
                d_loss_fake = discriminator.train_on_batch([initial_recon_images, y_batch], fake_labels)

                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                print("d_loss:{}".format(d_loss))

                """
                Train the generator network
                """

                z_noise2 = np.random.normal(0, 1, size=(batch_size, z_shape))
                random_labels = np.random.randint(0, 6, batch_size).reshape(-1, 1)
                random_labels = to_categorical(random_labels, 6)

                g_loss = adversarial_model.train_on_batch([z_noise2, random_labels], [1] * batch_size)

                print("g_loss:{}".format(g_loss))

                gen_losses.append(g_loss)
                dis_losses.append(d_loss)

            # Write losses to Tensorboard
            write_log(tensorboard, 'g_loss', np.mean(gen_losses), epoch)
            write_log(tensorboard, 'd_loss', np.mean(dis_losses), epoch)

            """
            Generate images after every 10th epoch
            """
            if epoch % 10 == 0:
                images_batch = loaded_images[0:batch_size]
                images_batch = images_batch / 127.5 - 1.0
                images_batch = images_batch.astype(np.float32)

                y_batch = y[0:batch_size]
                z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))

                gen_images = generator.predict_on_batch([z_noise, y_batch])

                for i, img in enumerate(gen_images[:5]):
                    save_rgb_img(img, path="results/img_{}_{}.png".format(epoch, i))

        # Save networks
        try:
            generator.save_weights("generator.h5")
            discriminator.save_weights("discriminator.h5")
        except Exception as e:
            print("Error:", e)

    """
    Train encoder
    """

    if TRAIN_ENCODER:
        # Build and compile encoder
        encoder = build_encoder(input_shape=image_shape)
        encoder.compile(loss=euclidean_distance_loss, optimizer='adam')

        # Load the generator network's weights
        try:
            generator.load_weights(os.path.join(saved_models_path, "generator.h5"))
        except Exception as e:
            print("Error:", e)

        z_i = np.random.normal(0, 1, size=(5000, z_shape))

        y = np.random.randint(low=0, high=6, size=(5000,), dtype=np.int64)
        num_classes = len(set(y))
        y = np.reshape(np.array(y), [len(y), 1])
        y = to_categorical(y, num_classes=num_classes)

        for epoch in range(epochs):
            print("Epoch:", epoch)

            encoder_losses = []

            number_of_batches = int(z_i.shape[0] / batch_size)
            print("Number of batches:", number_of_batches)
            for index in range(number_of_batches):
                print("Batch:", index + 1)

                z_batch = z_i[index * batch_size:(index + 1) * batch_size]
                y_batch = y[index * batch_size:(index + 1) * batch_size]

                generated_images = generator.predict_on_batch([z_batch, y_batch])

                # Train the encoder model
                encoder_loss = encoder.train_on_batch(generated_images, z_batch)
                print("Encoder loss:", encoder_loss)

                encoder_losses.append(encoder_loss)

            # Write the encoder loss to Tensorboard
            write_log(tensorboard, "encoder_loss", np.mean(encoder_losses), epoch)

        # Save the encoder model
        encoder.save_weights(os.path.join(saved_models_path, "encoder.h5"))

    """
    Optimize the encoder and the generator network
    """
    if TRAIN_GAN_WITH_FR:

        # Load the encoder network
        encoder = build_encoder()
        encoder.load_weights(os.path.join(saved_models_path, "encoder.h5"))

        # Load the generator network
        generator.load_weights(os.path.join(saved_models_path, "generator.h5"))

        image_resizer = build_image_resizer()
        image_resizer.compile(loss=['binary_crossentropy'], optimizer='adam')

        # Face recognition model
        fr_model = build_fr_model(input_shape=fr_image_shape)
        fr_model.compile(loss=['binary_crossentropy'], optimizer="adam")

        # Make the face recognition network as non-trainable
        fr_model.trainable = False

        # Input layers
        input_image = Input(shape=(64, 64, 3))
        input_label = Input(shape=(6,))

        # Use the encoder and the generator network
        latent0 = encoder(input_image)
        gen_images = generator([latent0, input_label])

        # Resize images to the desired shape
        resized_images = Lambda(lambda x: K.resize_images(gen_images, height_factor=3, width_factor=3,
                                                          data_format='channels_last'))(gen_images)
        embeddings = fr_model(resized_images)

        # Create a Keras model and specify the inputs and outputs for the network
        fr_adversarial_model = Model(inputs=[input_image, input_label], outputs=[embeddings])

        # Compile the model
        fr_adversarial_model.compile(loss=euclidean_distance_loss, optimizer=adversarial_optimizer)

        for epoch in range(epochs):
            print("Epoch:", epoch)

            reconstruction_losses = []

            number_of_batches = int(len(loaded_images) / batch_size)
            print("Number of batches:", number_of_batches)
            for index in range(number_of_batches):
                print("Batch:", index + 1)

                images_batch = loaded_images[index * batch_size:(index + 1) * batch_size]
                images_batch = images_batch / 127.5 - 1.0
                images_batch = images_batch.astype(np.float32)

                y_batch = y[index * batch_size:(index + 1) * batch_size]

                images_batch_resized = image_resizer.predict_on_batch(images_batch)

                real_embeddings = fr_model.predict_on_batch(images_batch_resized)

                reconstruction_loss = fr_adversarial_model.train_on_batch([images_batch, y_batch], real_embeddings)

                print("Reconstruction loss:", reconstruction_loss)

                reconstruction_losses.append(reconstruction_loss)

            # Write the reconstruction loss to Tensorboard
            write_log(tensorboard, "reconstruction_loss", np.mean(reconstruction_losses), epoch)

            """
            Generate images
            """
            if epoch % 10 == 0:
                images_batch = loaded_images[0:batch_size]
                images_batch = images_batch / 127.5 - 1.0
                images_batch = images_batch.astype(np.float32)

                y_batch = y[0:batch_size]
                z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))

                gen_images = generator.predict_on_batch([z_noise, y_batch])

                for i, img in enumerate(gen_images[:5]):
                    output_path = os.path.join(results_path, "img_opt_{}_{}.png".format(epoch, i))
                    save_rgb_img(img, path=output_path)

        # Save improved weights for both of the networks
        generator.save_weights(os.path.join(saved_models_path, "generator_optimized.h5"))
        encoder.save_weights(os.path.join(saved_models_path, "encoder_optimized.h5"))

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
