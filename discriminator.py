from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU, Conv2D, Concatenate

class Discriminator:
    def __init__(self, image_size, latent_code_length):
        self.latent_code_length = latent_code_length;
        self.image_size = image_size

    def build_model(self):
        x = Input(self.image_size)
        z = Input(self.latent_code_length)
        _z = Flatten()(z)
        _z = Dense(self.image_size[0]*self.image_size[1]*self.image_size[2])(_z)
        _z = Reshape(self.image_size)(_z)

        y = Concatenate()([x,_z])
        y = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(y)
        y = LeakyReLU()(y)
        y = Conv2D(128, (3, 3), padding="same")(y)
        y = LeakyReLU()(y)
        y = Conv2D(256, (3, 3), strides=(2, 2), padding="same")(y)
        y = LeakyReLU()(y)
        y = Conv2D(256, (3, 3), padding="same")(y)
        y = LeakyReLU()(y)
        y = Conv2D(512, (3, 3), strides=(2, 2), padding="same")(y)
        y = LeakyReLU()(y)
        y = Conv2D(512, (3, 3), padding="same")(y)
        y = LeakyReLU()(y)
        y = Conv2D(1024, (3, 3), strides=(2, 2), padding="same")(y)
        y = LeakyReLU()(y)
        y = Conv2D(1024, (3, 3), padding="same")(y)
        y = LeakyReLU()(y)
        y = Flatten()(y)
        y = Dense(1)(y)
        return Model([x, z], [y])
