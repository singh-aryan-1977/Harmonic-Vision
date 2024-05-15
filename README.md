# Project Name: Seeing Sound
Original Team Name: Harmonic-Vision
Team Member Names: Adithya Sriram, Arjan Chakravarthy, Aryan Singh

Our project seeks to create videos that change according to different features in an input audio file. We train our own GAN model on a few different datasets and analyze the output results from each of these outputs. The link to the google drive below contains our results and checkpoints of models used to create them.

Link to Google Drive: https://drive.google.com/drive/folders/1kT_zQcVXeL8rFwmanp2ePZ7CgC7SaWhD

To train GAN: python train_gan.py
To generate music video: python music.py --song SONG
Note: Data path to checkpoint.pth file will need to updated in the music.py file and name of dataset, datapath need to be updated in train_gan.py
