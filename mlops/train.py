import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Reshape, LeakyReLU, Conv2D, Flatten, Dropout, Conv2DTranspose
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

data_path = os.path.join(os.path.dirname(__file__), "MNIST.csv")
df = pd.read_csv(data_path, header=0)
labels = df.iloc[:, 0]
images = df.iloc[:, 1:].values
images = images.reshape(-1, 8, 8, 1).astype("float32")
images = (images - 8) / 8


def build_generator():
    model = Sequential()

    model.add(Dense(2 * 2 * 256, use_bias=False, input_shape=(100,)))
    model.add(LeakyReLU())
    model.add(Reshape((2, 2, 256)))

    model.add(
        Conv2DTranspose(
            128, (5, 5), strides=(
                2, 2), padding="same", use_bias=False))
    model.add(LeakyReLU())

    model.add(
        Conv2DTranspose(
            64, (5, 5), strides=(
                2, 2), padding="same", use_bias=False))
    model.add(LeakyReLU())

    model.add(
        Conv2D(
            1,
            (5,
             5),
            padding="same",
            use_bias=False,
            activation="tanh"))

    return model


generator = build_generator()
generator.summary()


def build_discriminator():
    model = Sequential()

    model.add(
        Conv2D(
            64, (3, 3), strides=(
                1, 1), padding="same", input_shape=[
                8, 8, 1]))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    return model


discriminator = build_discriminator()
discriminator.summary()

cross_entropy = tf.keras.losses.BinaryCrossentropy()


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = Adam(1e-4)
discriminator_optimizer = Adam(1e-4)

EPOCHS = 50
BATCH_SIZE = 128
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])


@tf.function
def train_step(images):
    current_batch_size = tf.shape(images)[0]
    noise = tf.random.normal([current_batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )
    return disc_loss, gen_loss


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")

    fig.savefig(f"image_at_epoch_{epoch:04d}.png")
    plt.close(fig)


dataset = (tf.data.Dataset.from_tensor_slices(
    images).shuffle(len(images)).batch(BATCH_SIZE))

for epoch in range(EPOCHS):
    disc_losses = []
    gen_losses = []
    for image_batch in dataset:
        disc_loss, gen_loss = train_step(image_batch)
        disc_losses.append(disc_loss.numpy())
        gen_losses.append(gen_loss.numpy())

    print(
        f"Epoch {epoch + 1}, Discriminator Loss: {np.mean(disc_losses):.4f}, "
        f"Generator Loss: {np.mean(gen_losses):.4f}"
    )

    if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
        generate_and_save_images(generator, epoch + 1, seed)

test_noise = tf.random.normal([BATCH_SIZE, noise_dim])
generated_images = generator(test_noise, training=False)

real_labels = tf.ones((BATCH_SIZE, 1))
fake_labels = tf.zeros((BATCH_SIZE, 1))

real_predictions = discriminator(images[:BATCH_SIZE], training=False)
fake_predictions = discriminator(generated_images, training=False)

real_accuracy = tf.reduce_mean(
    tf.cast(tf.greater_equal(real_predictions, 0.5), tf.float32)
).numpy()
fake_accuracy = tf.reduce_mean(
    tf.cast(tf.less(fake_predictions, 0.5), tf.float32)
).numpy()

print(f"\nFinal Discriminator Accuracy on Real Images: {real_accuracy:.4f}")
print(f"Final Discriminator Accuracy on Fake Images: {fake_accuracy:.4f}")
print(
    f"Average Discriminator Accuracy: "
    f"{(real_accuracy + fake_accuracy) / 2:.4f}")