# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:07:41 2019

@author: manan
"""
# import a few libraries

import scipy
from scipy import signal
from scipy.io.wavfile import write
import numpy as np

# multiplying the absolute value of the signal with the complex exponential of the phase to get the complete complex signal

comp_predicted_n = np.multiply(np.abs(predicted_n_adjusted.T), np.exp(1j*(Sl_n_ph)))/10 #(5000*np.abs(Sl_n.T))

# taking the inverse fourier transform of the complex signal, to reconstruct the signal

z,xr = scipy.signal.istft(comp_predicted_n, 16000, window='hann', nperseg=256, noverlap=32, nfft=512)

xr = np.float32(xr)

# writing the reconstructed signal to a file, for playing

write('reconstructed_150.wav', 16000, xr)

# plotting the reconstructed signal, the noisy signal and the original signal for comparison.

plt.plot(z, xr/5)

sl_temp = np.int(l.shape[0])

plt.plot(z[0:sl_temp], l)

l_noisy = np.float32(l_noisy)

plt.plot(z[0:sl_temp], l_noisy)

# writing the noisy and the original signal to files, for playing

write('noisy_150.wav', 16000, l_noisy)

write('original_150.wav', 16000, l)

#ysr, yt = read('reconstructed_150.mp3', normalized=True)

#scipy.io.wavfile.write('music_150.wav', 16000, xr)
