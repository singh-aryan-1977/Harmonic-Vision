import torch

from src.training_utils import training_utils
from src.model import discriminators
from src.model import generators


class GAN(torch.nn.Module):
    def generate_imgs(self, cls=None, noise=None, fixed=False):
        if fixed:
            noise = self.fixed_noise
        elif noise is None:
            noise = training_utils.truncated_normal(self.fixed_noise.shape).to(device=self.fixed_noise.device)
        cls = self.cls if cls is None else cls
        img_gen = self.generator(z=noise, cls=cls)
        return img_gen, noise


class BigGAN(GAN):
    def __init__(self, generator, discriminator, fixed_noise):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.fixed_noise = fixed_noise
        self.cls = None

    def forward(self, img_real, img_gen, cls):
        _, img_real_score = self.img_discriminator(x=img_real, cls=cls)
        _, img_gen_score = self.img_discriminator(x=img_gen, cls=cls)

        output = {
            "img_real_score": img_real_score,
            "img_gen_score": img_gen_score,
        }

        if self.cls is None: self.cls = cls.detach()
        return output

    def req_grad_disc(self, req_grad):
        for p in self.discriminator.parameters():
            p.requires_grad = req_grad

    def get_disc_params(self):
        return self.discriminator.parameters()

    def get_gen_params(self):
        return self.generator.parameters()

    @classmethod
    def from_config(cls, config):
        fixed_noise = training_utils.truncated_normal((config["bs"], config["latent_dim"]))
        return cls(
            generator=generators.GenBigGAN.from_config(config),
            discriminator=discriminators.DiscBigGAN.from_config(config),
            fixed_noise=fixed_noise.to(config["device"]),
        )
