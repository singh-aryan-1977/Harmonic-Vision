import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Concatenate,Flatten,Reshape,Conv2D,Conv2DTranspose,LeakyReLU,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
import numpy as np
from generator import Generator
from encoder import Encoder
from discriminator import Discriminator
from PIL import Image

def build_generator(image_size, latent_code_length, noise_vector):
    x = Input(latent_code_length)
    y = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same")(x)
    y = LeakyReLU()(y)
    y = Conv2D(512, (3, 3), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2D(256, (3, 3), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2D(128, (3, 3), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2DTranspose(image_size[-1],(3,3),strides=(2,2),padding="same")(y)
    return Model(x, y)

# def build_generator(image_size, latent_code_length, noise_vector):
#     # print(latent_code_length)
#     # print(noise_vector)
#     # noise_vector = Input(latent_code_length)
#     # noise_input = tf.keras.layers.Input(shape=(latent_code_length,)) # just have it for now like this
#     return Generator(image_size=image_size).build_model(noise_input=noise_vector) # when we have our own noise_vector, just pass it in here

# def build_encoder(image_size, latent_code_length):
#     x = Input(image_size)
#     y = Conv2D(64, (4, 4), strides=(2, 2), padding="same")(x)
#     y = LeakyReLU()(y)
#     y = Conv2D(128, (4, 4), strides=(2, 2), padding="same")(y)
#     y = LeakyReLU()(y)
#     y = Conv2D(256, (4, 4), strides=(2, 2), padding="same")(y)
#     y = LeakyReLU()(y)
#     y = Flatten()(y)
#     y = Dense(np.prod(latent_code_length))(y)
#     y = Reshape(latent_code_length)(y)
#     return Model(x, y)

def build_encoder(image_size, latent_code_length):
    return Encoder(image_size=image_size, latent_code_length=latent_code_length).build_model()

# def build_discriminator(image_size, latent_code_length):
#     x = Input(image_size)
#     z = Input(latent_code_length)
#     _z = Flatten()(z)
#     _z = Dense(image_size[0]*image_size[1]*image_size[2])(_z)
#     _z = Reshape(image_size)(_z)

#     y = Concatenate()([x,_z])
#     y = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(y)
#     y = LeakyReLU()(y)
#     y = Conv2D(128, (3, 3), padding="same")(y)
#     y = LeakyReLU()(y)
#     y = Conv2D(256, (3, 3), strides=(2, 2), padding="same")(y)
#     y = LeakyReLU()(y)
#     y = Conv2D(256, (3, 3), padding="same")(y)
#     y = LeakyReLU()(y)
#     y = Conv2D(512, (3, 3), strides=(2, 2), padding="same")(y)
#     y = LeakyReLU()(y)
#     y = Conv2D(512, (3, 3), padding="same")(y)
#     y = LeakyReLU()(y)
#     y = Conv2D(1024, (3, 3), strides=(2, 2), padding="same")(y)
#     y = LeakyReLU()(y)
#     y = Conv2D(1024, (3, 3), padding="same")(y)
#     y = LeakyReLU()(y)
#     y = Flatten()(y)
#     y = Dense(1)(y)
#     return Model([x, z], [y])
def build_discriminator(image_size, latent_code_length):
    return Discriminator(image_size=image_size, latent_code_length=latent_code_length).build_model()

def build_train_step(generator, encoder, discriminator):
    # Define optimizers
    g_optimizer = Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.9)
    e_optimizer = Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.9)
    d_optimizer = Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.9)
    @tf.function
    def train_step(real_image, real_code, noise_vector):
        with tf.GradientTape() as tape:
            fake_image = generator(real_code)
            fake_code = encoder(real_image)

            d_inputs = [fake_image, real_image]
            d_preds = discriminator(d_inputs)
            pred_g, pred_e = tf.split(d_preds, num_or_size_splits=2, axis=0)

            d_loss = tf.reduce_mean(tf.nn.softplus(pred_g)) + \
                     tf.reduce_mean(tf.nn.softplus(-pred_e))
            g_loss = tf.reduce_mean(tf.nn.softplus(-pred_g))
            e_loss = tf.reduce_mean(tf.nn.softplus(pred_e))

        d_gradients = tape.gradient(d_loss, discriminator.trainable_variables)
        g_gradients = tape.gradient(g_loss, generator.trainable_variables)
        e_gradients = tape.gradient(e_loss, encoder.trainable_variables)

        d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
        g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
        e_optimizer.apply_gradients(zip(e_gradients, encoder.trainable_variables))

        return d_loss, g_loss, e_loss

    return train_step

def train():
    check_point = 500
    iters = 200 * check_point
    image_size = (128, 128, 3)  # Adjusted image size
    latent_code_length = (4, 4, 64)  # Adjusted latent code length
    batch_size = 16

    (x_train, _), (_, _) = cifar10.load_data()
    x_train = (x_train.astype("float32") / 127.5) - 1.0

    num_of_data = x_train.shape[0]
    z_train = np.random.uniform(-1.0, 1.0, (num_of_data,) + latent_code_length).astype("float32")


    noise_input = Input(latent_code_length)
    print("latent code length")
    print(latent_code_length)
    generator = build_generator(image_size, latent_code_length, noise_input)
    encoder = build_encoder(image_size, latent_code_length)
    discriminator = build_discriminator(image_size, latent_code_length)
    train_step = build_train_step(generator, encoder, discriminator)

    for i in range(iters):
        real_images = x_train[np.random.randint(0, num_of_data, batch_size)]
        real_code = z_train[np.random.randint(0, num_of_data, batch_size)]
        noise_vector = np.random.normal(size=(batch_size,) + latent_code_length).astype("float32")

        d_loss, g_loss, e_loss = train_step(real_images, real_code, noise_vector)
        print("\r[{}/{}]  d_loss: {:.4}, g_loss: {:.4}, e_loss: {:.4}".format(i, iters, d_loss, g_loss, e_loss),
              end="")

        if (i + 1) % check_point == 0:
            # Save G(x) image
            generated_image = generator.predict(encoder.predict(x_train[:1]))
            generated_image = np.clip(0.5 * generated_image + 0.5, 0, 1)  # Convert back to [0,1] range
            generated_image = (generated_image * 255).astype(np.uint8)
            Image.fromarray(generated_image[0]).save("G_E_x-{}.png".format(i // check_point))

            # Save G(z) image
            generated_image = generator.predict(np.random.uniform(-1, 1, (1,) + latent_code_length))
            generated_image = np.clip(0.5 * generated_image + 0.5, 0, 1)  # Convert back to [0,1] range
            generated_image = (generated_image * 255).astype(np.uint8)
            Image.fromarray(generated_image[0]).save("G_z-{}.png".format(i // check_point))

if __name__ == "__main__":
    train()