from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, Flatten
import numpy as np

class Encoder:
    def __init__(self, image_size, latent_code_length):
        self.image_size = image_size
        self.latent_code_length = latent_code_length;

    def build_model(self):
        x = Input(self.image_size)
        y = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(x)
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
        y = Conv2D(self.latent_code_length[-1],(3,3),strides=(2,2),padding="same")(y)
        print("reach here")
        return Model(x, y)