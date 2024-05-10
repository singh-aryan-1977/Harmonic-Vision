from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2DTranspose, LeakyReLU, Conv2D

class Generator:
    def __init__(self, image_size):
        self.image_size = image_size

    def build_model(self, noise_input):
        # print("noise input")
        # print(noise_input)
        # logits = Dense(8 * 8 * 512)(noise_input)
        # print("logits")
        # print(logits)

        # input = Input(latent_code_length)
        output_tensor = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same")(noise_input)
        output_tensor = LeakyReLU()(output_tensor)
        output_tensor = Conv2D(512, (3, 3), padding="same")(output_tensor)
        output_tensor = LeakyReLU()(output_tensor)
        output_tensor = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(output_tensor)
        output_tensor = LeakyReLU()(output_tensor)
        output_tensor = Conv2D(256, (3, 3), padding="same")(output_tensor)
        output_tensor = LeakyReLU()(output_tensor)
        output_tensor = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(output_tensor)
        output_tensor = LeakyReLU()(output_tensor)
        output_tensor = Conv2D(128, (3, 3), padding="same")(output_tensor)
        output_tensor = LeakyReLU()(output_tensor)
        output_tensor = Conv2DTranspose(self.image_size[-1],(3,3),strides=(2,2),padding="same")(output_tensor)
        return Model(noise_input, output_tensor)
