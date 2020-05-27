# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:27:22 2019

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

from tensorflow import keras

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

import pandas as pd

# Function to read mp3 files as numpy array

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

X = [None]*999 # initialize a list in which to put the numpy arrays from the common-voice files

# A for-loop to read the common-voice files

for i in range(0,999):
    sr, x = read('common_voice_en ({}).mp3'.format(i+1), normalized=True)
    x = librosa.core.resample(x, sr, 16000)
    X[i] = x
    

S = [None]*999    # initialize a list in which to put the spectrograms in

# Also initialize lists for the time stamps and frequencies

t = [None]*999
f = [None]*999

# Take the spectrograms of all the files using a for-loop

for i in range(0,999):
    f[i], t[i], S[i] = scipy.signal.stft(X[i], 16000, window='hann', nperseg=256, noverlap=32, nfft=512)
    
# convert the list into a numpy array of the spectrograms which will be fed into the autoencoder

S_test = np.concatenate(S, axis=1 )

S_usable = np.abs(S_test)

S_phase = np.angle(S_test) # saving phase for reconstruction later on

S_usable = S_usable.T # S_usable is in a form which can be fed into the autoencoder

## Autoencoder in Keras

input_img= Input(shape=(257,))

# encoded and decoded layer for the autoencoder

encoded = Dense(units=128, activation='relu')(input_img)
encoded = Dense(units=64, activation='relu')(encoded)
encoded = Dense(units=32, activation='relu')(encoded)
decoded = Dense(units=64, activation='relu')(encoded)
decoded = Dense(units=128, activation='relu')(decoded)
decoded = Dense(units=257, activation='sigmoid')(decoded)

# Building autoencoder

autoencoder=Model(input_img, decoded)

#extracting encoder

encoder = Model(input_img, encoded)

# compiling the autoencoder

autoencoder.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the noise trained data to the autoencoder
 
autoencoder.fit(S_usable, S_usable,
                epochs=100,
                batch_size=256,
                shuffle=True)

#Save the autoencoder to a file

autoencoder.save('AE_1000.hdf5', overwrite=True, include_optimizer=True)

# Feeding a particular file into the autoencoder and getting the output

Temp1 = np.abs(S[2])
Temp = Temp1.T
encoded_imgs = encoder.predict(Temp)
predicted = autoencoder.predict(Temp)

# making adjustment to the prediction

predicted_scaled = np.multiply(100000*predicted, Temp)

# plotting the adjusted autoencoder output, and the original input

plt.pcolormesh(t[2], f[2], predicted_scaled.T)
plt.colorbar()
plt.clim(0, 0.05)
plt.show()

plt.pcolormesh(t[2], f[2], 10*Temp1)
plt.colorbar()
plt.clim(0, 0.05)
plt.show()

