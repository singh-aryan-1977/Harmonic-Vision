import librosa
import argparse
import numpy as np
import moviepy.editor as mpy
import random
import torch
from PIL import Image
from tqdm import tqdm
# from pytorch_pretrained_biggan import (one_hot_from_names, save_as_images, display_in_terminal, convert_to_images)
from scipy.stats import truncnorm

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

sys.path.append(parent_dir)

from src.pipeline.pipeline import GANPipeline
from src.training_utils.training_utils import get_config

#get input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--song",required=True)
parser.add_argument("--resolution", default='512')
parser.add_argument("--duration", type=int)
parser.add_argument("--pitch_sensitivity", type=int, default=220)
parser.add_argument("--tempo_sensitivity", type=float, default=0.25)
parser.add_argument("--depth", type=float, default=1)
parser.add_argument("--classes", nargs='+', type=int)
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--sort_classes_by_power", type=int, default=0)
parser.add_argument("--jitter", type=float, default=0.5)
parser.add_argument("--frame_length", type=int, default=512)
parser.add_argument("--truncation", type=float, default=1)
parser.add_argument("--smooth_factor", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=30)
parser.add_argument("--use_previous_classes", type=int, default=0)
parser.add_argument("--use_previous_vectors", type=int, default=0)
parser.add_argument("--output_file", default="output.mp4")
args = parser.parse_known_args()

def truncated_noise_sample(batch_size=1, dim_z=100, truncation=1., seed=None):
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return truncation * values

#read song
args = args[0]
if args.song:
    song=args.song
    print('\nReading audio \n')
    y, sr = librosa.load(song)
else:
    raise ValueError("you must enter an audio file name in the --song argument")

#set model name based on resolution
model_name='biggan-deep-' + args.resolution

frame_length=args.frame_length

#set pitch sensitivity
pitch_sensitivity=(300-args.pitch_sensitivity) * 512 / frame_length

#set tempo sensitivity
tempo_sensitivity=args.tempo_sensitivity * frame_length / 512

#set depth
depth=args.depth

#set number of classes
num_classes=args.num_classes

#set sort_classes_by_power
sort_classes_by_power=args.sort_classes_by_power

#set jitter
jitter=args.jitter

#set truncation
truncation=args.truncation

#set batch size
batch_size=args.batch_size

#set use_previous_classes
use_previous_vectors=args.use_previous_vectors

#set use_previous_vectors
use_previous_classes=args.use_previous_classes

#set output name
outname=args.output_file

#set smooth factor
if args.smooth_factor > 1:
    smooth_factor=int(args.smooth_factor * 512 / frame_length)
else:
    smooth_factor=args.smooth_factor

#set duration
if args.duration:
    seconds=args.duration
    frame_lim=int(np.floor(seconds*22050/frame_length/batch_size))
else:
    frame_lim=int(np.floor(len(y)/sr*22050/frame_length/batch_size))



# Load pre-trained model
checkpoint = '/Users/adithyasriram/Desktop/CSCI 1430/Harmonic-Vision/checkpoints/checkpoint_150.pth' 
state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))

config = {
        'ds_name': 'CIFAR10',
        'num_cls': 10,
        'loading_normalization_mean': 0.5,
        'loading_normalization_var': 0.5,
        'w_init': torch.nn.init.orthogonal_,
        'save_metric_interval': 1,
        'logging_interval': 10,
        'seed': 420,
        'device': 'cuda',
        'img_rows': 4,
        'save_img_count': 12,
        'real_imgs_save_path': './data/{ds_name}/{model_architecture}/real_img/{hparams}',
        'gen_imgs_save_path': './data/{ds_name}/{model_architecture}/gen_img/{hparams}',
        'logging_path': './data/{ds_name}/{model_architecture}/logs/{name}',
        'save_model_path': './data/{ds_name}/{model_architecture}/checkpoints/{hparams}',
        'save_name': 'gan',
        'save_model_interval': 50,
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
            'pre': (1024, 512),
            'post': (512,), 'colors': 3
            },
        'disc_mult_chs': {
            'colors': 3,
            'pre': (512,),
            'post': (512, 1024)
            },
        'enc_mult_chs': {
            'colors': 3,
            'blocks': (64, 64, 128, 128, 256, 256, 512, 512)
            },
        'enc_hidden': 256,
        'enc_in_mlp_dim': 512,
        'ks': 3,
        'image_size': 32,
        'latent_disc_blocks': 5,
        'latent_disc_mlp_dim': 128,
        'comb_disc_blocks': 5,
        'comb_disc_mlp_dim': 128,
        'embedding_dim': 64,
        'latent_dim': 100,
        'enc_out_dim': 100,
        'bs': 128,
        'model_architecture': 'biggan',
        'hparams_str': ''
    }

data_path = '../input/cifar10-dataset'

pipeline = GANPipeline.from_config(data_path, config)

pipeline.model.load_state_dict(state_dict)
pipeline.model.eval()

model = pipeline.model

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


########################################
########################################
########################################
########################################
########################################


#create spectrogram
spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=100,fmax=8000, hop_length=frame_length)

#get mean power at each time point
specm=np.mean(spec,axis=0)

#compute power gradient across time points
gradm=np.gradient(specm)

#set max to 1
gradm=gradm/np.max(gradm)

#set negative gradient time points to zero
gradm = gradm.clip(min=0)

#normalize mean power between 0-1
specm=(specm-np.min(specm))/np.ptp(specm)

#create chromagram of pitches X time points
chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=frame_length)

#sort pitches by overall power
chromasort=np.argsort(np.mean(chroma,axis=1))[::-1]



########################################
########################################
########################################
########################################
########################################


if args.classes:
    classes=args.classes
    if len(classes) not in [12,num_classes]:
        raise ValueError("The number of classes entered in the --class argument must equal 12 or [num_classes] if specified")

elif args.use_previous_classes==1:
    cvs=np.load('class_vectors.npy')
    classes=list(np.where(cvs[0]>0)[0])

else: #select 12 random classes
    cls1000=list(range(1000))
    random.shuffle(cls1000)
    classes=cls1000[:12]


classes = list(range(10))


if sort_classes_by_power==1:
    classes=[classes[s] for s in np.argsort(chromasort[:num_classes])]



#initialize first class vector
cv1=np.zeros(10)
for pi,p in enumerate(chromasort[:num_classes]):

    if num_classes < 12:
        cv1[classes[pi]] = chroma[p][np.min([np.where(chrow>0)[0][0] for chrow in chroma])]
    else:
        cv1[classes[p]] = chroma[p][np.min([np.where(chrow>0)[0][0] for chrow in chroma])]

#initialize first noise vector
nv1 = truncated_noise_sample(truncation=truncation)[0]

#initialize list of class and noise vectors
class_vectors=[cv1]
noise_vectors=[nv1]

#initialize previous vectors (will be used to track the previous frame)
cvlast=cv1
nvlast=nv1


#initialize the direction of noise vector unit updates
update_dir=np.zeros(100)
for ni,n in enumerate(nv1):
    if n<0:
        update_dir[ni] = 1
    else:
        update_dir[ni] = -1


#initialize noise unit update
update_last=np.zeros(100)


########################################
########################################
########################################
########################################
########################################


#get new jitters
def new_jitters(jitter):
    jitters=np.zeros(100)
    for j in range(100):
        if random.uniform(0,1)<0.5:
            jitters[j]=1
        else:
            jitters[j]=1-jitter
    return jitters


#get new update directions
def new_update_dir(nv2,update_dir):
    for ni,n in enumerate(nv2):
        if n >= 2*truncation - tempo_sensitivity:
            update_dir[ni] = -1

        elif n < -2*truncation + tempo_sensitivity:
            update_dir[ni] = 1
    return update_dir


#smooth class vectors
def smooth(class_vectors,smooth_factor):

    if smooth_factor==1:
        return class_vectors

    class_vectors_terp=[]
    for c in range(int(np.floor(len(class_vectors)/smooth_factor)-1)):
        ci=c*smooth_factor
        cva=np.mean(class_vectors[int(ci):int(ci)+smooth_factor],axis=0)
        cvb=np.mean(class_vectors[int(ci)+smooth_factor:int(ci)+smooth_factor*2],axis=0)

        for j in range(smooth_factor):
            cvc = cva*(1-j/(smooth_factor-1)) + cvb*(j/(smooth_factor-1))
            class_vectors_terp.append(cvc)

    return np.array(class_vectors_terp)


#normalize class vector between 0-1
def normalize_cv(cv2):
    min_class_val = min(i for i in cv2 if i != 0)
    for ci,c in enumerate(cv2):
        if c==0:
            cv2[ci]=min_class_val
    cv2=(cv2-min_class_val)/np.ptp(cv2)

    return cv2


print('\nGenerating input vectors \n')

for i in tqdm(range(len(gradm))):

    #print progress
    pass

    #update jitter vector every 100 frames by setting ~half of noise vector units to lower sensitivity
    if i%200==0:
        jitters=new_jitters(jitter)

    #get last noise vector
    nv1=nvlast

    #set noise vector update based on direction, sensitivity, jitter, and combination of overall power and gradient of power
    update = np.array([tempo_sensitivity for k in range(100)]) * (gradm[i]+specm[i]) * update_dir * jitters

    #smooth the update with the previous update (to avoid overly sharp frame transitions)
    update=(update+update_last*3)/4

    #set last update
    update_last=update

    #update noise vector
    nv2=nv1+update

    #append to noise vectors
    noise_vectors.append(nv2)

    #set last noise vector
    nvlast=nv2

    #update the direction of noise units
    update_dir=new_update_dir(nv2,update_dir)

    #get last class vector
    cv1=cvlast

    #generate new class vector
    cv2=np.zeros(10)
    for j in range(num_classes):

        cv2[classes[j]] = (cvlast[classes[j]] + ((chroma[chromasort[j]][i])/(pitch_sensitivity)))/(1+(1/((pitch_sensitivity))))

    #if more than 6 classes, normalize new class vector between 0 and 1, else simply set max class val to 1
    if num_classes > 6:
        cv2=normalize_cv(cv2)
    else:
        cv2=cv2/np.max(cv2)

    #adjust depth
    cv2=cv2*depth

    #this prevents rare bugs where all classes are the same value
    if np.std(cv2[np.where(cv2!=0)]) < 0.0000001:
        cv2[classes[0]]=cv2[classes[0]]+0.01

    #append new class vector
    class_vectors.append(cv2)

    #set last class vector
    cvlast=cv2


#interpolate between class vectors of bin size [smooth_factor] to smooth frames
class_vectors=smooth(class_vectors,smooth_factor)

#check whether to use vectors from last run
if use_previous_vectors==1:
    #load vectors from previous run
    class_vectors=np.load('class_vectors.npy')
    noise_vectors=np.load('noise_vectors.npy')
else:
    #save record of vectors for current video
    np.save('class_vectors.npy',class_vectors)
    np.save('noise_vectors.npy',noise_vectors)



########################################
########################################
########################################
########################################
########################################


#convert to Tensor
noise_vectors = torch.Tensor(np.array(noise_vectors))
class_vectors = torch.Tensor(np.array(class_vectors))

#Generate frames in batches of batch_size

print('\n\nGenerating frames \n')

#send to CUDA if running on GPU
model=model.to(device)
noise_vectors=noise_vectors.to(device)
class_vectors=class_vectors.to(device)


frames = []

for i in tqdm(range(frame_lim)):

    #print progress
    pass

    if (i+1)*batch_size > len(class_vectors):
        torch.cuda.empty_cache()
        break

    #get batch
    noise_vector=noise_vectors[i*batch_size:(i+1)*batch_size]
    class_vector=class_vectors[i*batch_size:(i+1)*batch_size]

    class_vector = torch.argmax(class_vector, dim=1)
    # Generate images
    with torch.no_grad():
        print(noise_vector.shape)
        print(class_vector.shape)
        output, _ = model.generate_imgs(cls=class_vector.long(), noise=noise_vector)

    output_cpu=output.cpu().data.numpy()

    #convert to image array and add to frames
    # for out in output_cpu:
    #     print(out.shape)
    for out in output_cpu:
        out = out.transpose((1, 2, 0))
        out = np.clip(((out + 1) / 2.0) * 256, 0, 255)
        im = []
        for i, o in enumerate(out):
          out_array = np.asarray(np.uint8(o), dtype=np.uint8)
          im.append(Image.fromarray(out_array))
        im=np.array(im)
        frames.append(im)

    #empty cuda cache
    torch.cuda.empty_cache()



#Save video
aud = mpy.AudioFileClip(song, fps = 44100)

if args.duration:
    aud.duration=args.duration

clip = mpy.ImageSequenceClip(frames, fps=22050/frame_length)
clip = clip.set_audio(aud)
clip.write_videofile(outname,audio_codec='aac')
