import librosa
import argparse
import numpy as np
import moviepy.editor as mpy
import random
import torch
from PIL import Image
from tqdm import tqdm
from scipy.stats import truncnorm

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

sys.path.append(parent_dir)

from src.pipeline.pipeline import GANPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--song",required=True)
args = parser.parse_known_args()

def truncated_noise_sample(batch_size=1, dim_z=120, truncation=1., seed=None):
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return truncation * values

args = args[0]
if args.song:
    song = args.song
    print('\nReading audio \n')
    y, sr = librosa.load(song)
else:
    raise ValueError("you must enter an audio file name in the --song argument")

smooth_factor = 20

FRAME_LENGTH = 512
PITCH = 80 * 512 / FRAME_LENGTH
TEMPO = 0.25 * FRAME_LENGTH / 512
DEPTH = 1
NUM_CLASSES = 5
JITTER = 0.8
TRUNCATION = 1
BATCH_SIZE = 30
OUTPUT_PATH = "output.mp4"
DATA_PATH = '../input/custom_dataset'
CHECKPOINT_PATH = './data/custom/biggan2/checkpoints/checkpoint_225.pth'
SMOOTH_FACTOR = int((smooth_factor*512)/FRAME_LENGTH)

frame_lim = int(np.floor(len(y)/sr*22050/FRAME_LENGTH/BATCH_SIZE))

state_dict = torch.load(CHECKPOINT_PATH)

config = {
        'ds_name': 'custom',
        'num_cls': 5,
        'loading_normalization_mean': 0.5,
        'loading_normalization_var': 0.5,
        'w_init': None,
        'save_metric_interval': 10,
        'logging_interval': 35,
        'seed': 420,
        'device': 'cuda',
        'img_rows': 4,
        'save_img_count': 12,
        'real_imgs_save_path': './data/{ds_name}/{model_architecture}2/real_img/{hparams}',
        'gen_imgs_save_path': './data/{ds_name}/{model_architecture}2/gen_img/{hparams}',
        'logging_path': './data/{ds_name}/{model_architecture}2/logs/{name}',
        'save_model_path': './data/{ds_name}/{model_architecture}2/checkpoints/{hparams}',
        'save_name': 'gan',
        'save_model_interval': 25,
        'clf_lr': 0.0002,
        'disc_steps': 2,
        'gen_steps': 1,
        'epochs': 2000,
        'lr_gen': 0.0002,
        'lr_disc': 0.0002,
        'betas': (0.5, 0.999),
        'dropout': 0.2,
        'spectral_norm': True,
        'weight_cutoff': 0.0,
        'add_noise': 0,
        'gen_mult_chs': {
            'pre': (1024, 512, 256, 128),
            'post': (256,),
            'colors': 3
            },
        'disc_mult_chs': {
            'colors': 3,
            'pre': (256,),
            'post': (128, 256, 512, 1024)
            },
        'enc_mult_chs': {
            'colors': 3,
            'blocks': (128, 128, 256, 256, 512, 512, 1024, 1024)
            },
        'enc_hidden': 512,
        'enc_in_mlp_dim': 512,
        'ks': 3,
        'image_size': 128,
        'latent_disc_blocks': 6,
        'latent_disc_mlp_dim': 256,
        'comb_disc_blocks': 6,
        'comb_disc_mlp_dim': 256,
        'embedding_dim': 128,
        'latent_dim': 120,
        'enc_out_dim': 120,
        'bs': 16,
        'model_architecture': 'biggan',
        'hparams_str': ''
    }


pipeline = GANPipeline.from_config(DATA_PATH, config)

pipeline.model.load_state_dict(state_dict)
pipeline.model.eval()

model = pipeline.model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

spectogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=120, fmax=8000, hop_length=FRAME_LENGTH)

mean_spectogram = np.mean(spectogram, axis=0)

gradient = np.gradient(mean_spectogram)
gradient = (gradient/np.max(gradient)).clip(min=0)

mean_spectogram = (mean_spectogram-np.min(mean_spectogram))/np.ptp(mean_spectogram)

chromagram = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=FRAME_LENGTH)

chromasort = np.argsort(np.mean(chromagram, axis=1))[::-1]

classes = list(range(NUM_CLASSES))

init_class = np.zeros(NUM_CLASSES)
for index, c in enumerate(chromasort[:NUM_CLASSES]):
    if NUM_CLASSES < 12:
        init_class[classes[index]] = chromagram[c][np.min([np.where(chroma > 0)[0][0] for chroma in chromagram])]
    else:
        init_class[classes[c]] = chromagram[c][np.min([np.where(chroma > 0)[0][0] for chroma in chromagram])]

init_noise = truncated_noise_sample(truncation=TRUNCATION)[0]

class_vectors = [init_class]
noise_vectors = [init_noise]

prev_class = init_class
prev_noise = init_noise

direction = np.zeros(120)
for noise_index, noise in enumerate(init_noise):
    if noise < 0:
        direction[noise_index] = 1
    else:
        direction[noise_index] = -1


last_dir = np.zeros(120)

def generate_jitters(jitter):
    new_jitters = np.zeros(120)

    for j in range(120):
        if random.uniform(0,1) < 0.5:
            new_jitters[j] = 1
        else:
            new_jitters[j] = 1-jitter

    return new_jitters


def update_direction(noise_vec, dir_vec):
    for n_index, n_vector in enumerate(noise_vec):
        if n_vector >= 2*TRUNCATION - TEMPO:
            dir_vec[n_index] = -1

        elif n_vector < -2*TRUNCATION + TEMPO:
            dir_vec[n_index] = 1

    return dir_vec


def smooth(class_vec, smooth_factor):
    if smooth_factor == 1:
        return class_vec

    new_class_vec = []

    for i in range(int(np.floor(len(class_vec)/smooth_factor)-1)):
        smooth_class = i*smooth_factor
        class_1 = np.mean(class_vec[int(smooth_class):int(smooth_class)+smooth_factor], axis=0)
        class_2 = np.mean(class_vec[int(smooth_class)+smooth_factor:int(smooth_class)+smooth_factor*2], axis=0)

        for j in range(smooth_factor):
            new_class = class_1*(1-j/(smooth_factor-1)) + class_2*(j/(smooth_factor-1))
            new_class_vec.append(new_class)

    return np.array(new_class_vec)


def normalize_classes(class_vec):
    min_class_val = min(i for i in class_vec if i != 0)
    for class_ind, c in enumerate(class_vec):
        if c == 0:
            class_vec[class_ind] = min_class_val
    class_vec = (class_vec-min_class_val)/np.ptp(class_vec)

    return class_vec


print('\nGenerating input vectors \n')

for i in tqdm(range(len(gradient))):
    if i % 200 == 0:
        jitters = generate_jitters(JITTER)

    curr_noise = prev_noise

    update = np.array([TEMPO for _ in range(120)]) * (gradient[i]+mean_spectogram[i]) * direction * jitters

    update = (update + 3*last_dir)/4

    last_dir = update

    n_vec = curr_noise + update

    noise_vectors.append(n_vec)

    prev_noise = n_vec

    direction = update_direction(n_vec, direction)

    init_class = prev_class

    new_class_vec = np.zeros(NUM_CLASSES)
    for j in range(NUM_CLASSES):
        new_class_vec[classes[j]] = (prev_class[classes[j]] + ((chromagram[chromasort[j]][i])/(PITCH)))/(1+(1/((PITCH))))

    if NUM_CLASSES > 6:
        new_class_vec = normalize_classes(new_class_vec)
    else:
        new_class_vec = new_class_vec/np.max(new_class_vec)

    new_class_vec *= DEPTH

    if np.std(new_class_vec[np.where(new_class_vec != 0)]) < 0.0000001:
        new_class_vec[classes[0]] = new_class_vec[classes[0]] + 0.01

    class_vectors.append(new_class_vec)

    prev_class = new_class_vec


class_vectors = smooth(class_vectors, SMOOTH_FACTOR)

noise_vectors = torch.Tensor(np.array(noise_vectors))
class_vectors = torch.Tensor(np.array(class_vectors))


print('\n\nGenerating frames \n')

model = model.to(device)
noise_vectors = noise_vectors.to(device)
class_vectors = class_vectors.to(device)

video_frames = []

for i in tqdm(range(frame_lim)):

    if (i+1)*BATCH_SIZE > len(class_vectors):
        torch.cuda.empty_cache()
        break

    noise_vec = noise_vectors[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    class_vec = class_vectors[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

    class_vec = torch.argmax(class_vec, dim=1)

    with torch.no_grad():
        output, _ = model.generate_imgs(cls=class_vec.long(), noise=noise_vec)

    output_cpu = output.cpu().data.numpy()

    for res in output_cpu:
        res = res.transpose((1, 2, 0))
        res = np.clip(((res + 1) / 2.0) * 256, 0, 255)
        images = []
        for ind, out in enumerate(res):
          output_array = np.asarray(np.uint8(out), dtype=np.uint8)
          images.append(Image.fromarray(output_array))
        images = np.array(images)
        video_frames.append(images)

    torch.cuda.empty_cache()


audio = mpy.AudioFileClip(song, fps = 44100)

clip = mpy.ImageSequenceClip(video_frames, fps=22050/FRAME_LENGTH)
clip = clip.set_audio(audio)
clip.write_videofile(OUTPUT_PATH, audio_codec='aac')
