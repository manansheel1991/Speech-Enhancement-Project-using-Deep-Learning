# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:14:39 2019

@author: manan
"""
# Import necessary libraries

import pydub 
import scipy
import numpy as np
from pydub import AudioSegment
from scipy import signal
import librosa
from librosa import core
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from scipy.io.wavfile import write

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model

import pandas as pd

# function to read an mp3 file as a numpy array

def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y

# Reading data that is not in the autoencoder training set, and taking stft

srl, l = read('common_voice_en (1501).mp3', normalized=True)
l = librosa.core.resample(l, srl, 16000)
fl, tl, Sl = scipy.signal.stft(l, 16000, window='hann', nperseg=256, noverlap=32, nfft=512)

Sl_ph = np.angle(Sl)

# Adding noise to data and taking stft

#l_noisy = l + (tf.random.normal(tf.shape(l)))/10

l_noisy = l + np.random.normal(0, 0.1, np.shape(l))/5

#write('Noisy_data.wav', 16000, l_noisy)

fl_n, tl_n, Sl_n = scipy.signal.stft(l_noisy, 16000, window='hann', nperseg=256, noverlap=32, nfft=512)

Sl_n_ph = np.angle(Sl_n)

Sl_n = np.abs(Sl_n)

Sl_n = Sl_n.T

# Feeding noisy data into already trained autoencoder, which was trained in other code

autoencoder_loaded = load_model('AE_1000.hdf5')

predicted_n = autoencoder_loaded.predict(Sl_n)

# adjusting the output of the autoencoder

predicted_n_adjusted = np.multiply(5000*predicted_n, Sl_n)

#reconstruct_to_wav(np.int16(l_noisy*1000), predicted_n_adjusted*10, 'noisy_gaussian_1501.wav', 'reconstructed_gaussian_1501.wav', Sl_n_ph)


# plotting the noised image, the adjusted output of the autoencoder and the original file

plt.pcolormesh(tl, fl, predicted_n_adjusted.T)
plt.colorbar()
plt.clim(0, 0.05)
plt.show()

plt.pcolormesh(tl, fl, 10*Sl_n.T)
plt.colorbar()
plt.clim(0, 0.05)
plt.show()

plt.pcolormesh(tl, fl, 10*np.abs(Sl))
plt.colorbar()
plt.clim(0, 0.05)
plt.show()

