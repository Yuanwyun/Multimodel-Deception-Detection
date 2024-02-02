import librosa
import numpy as np
import torch
import os
import matplotlib.pyplot as plt


source_folder = "path"
target_folder = "path"

for i in os.listdir(source_folder):

    file_name = i.split('.wav')[0]


    # load audio
    y, sr = librosa.load(source_folder + i, sr=16000, mono=True)
    # ** high-resolution spec (whisper params)
    librosa_mel_spectrogram = librosa.feature.melspectrogram(
                                    y=y, 
                                    sr=sr,
                                    #n_fft=400,
                                    #hop_length=160,
                                    #win_length=400, 
                                    n_mels=128)

    # convert power to decibels
    librosa_mel_spectrogram = librosa.power_to_db(librosa_mel_spectrogram, ref=np.max)
    # load the above spectrogram as a torch tensor
    torch_mel_spectrogram = torch.tensor(np.asarray(librosa_mel_spectrogram))
    # standard normalization
    torch_mel_spectrogram = (torch_mel_spectrogram - torch_mel_spectrogram.mean()) / (torch_mel_spectrogram.std() + 1e-6)

    # apply sigmoid to squash the values between 0 to 1
    torch_mel_spectrogram = torch.sigmoid(torch_mel_spectrogram)
    # save the spec as a pytorch tensor
    torch.save(torch_mel_spectrogram, target_folder + file_name + '.pth')

    print(torch_mel_spectrogram.shape, torch_mel_spectrogram.min(),torch_mel_spectrogram.max())
    print(target_folder + file_name + '.pth')
    break
