# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:47:05 2024

@author: joukv
"""

import os
os.add_dll_directory("C:/Kabata/fCWT/libs")
from fcwt import *
import numpy as np
import matplotlib.pyplot as plt


#%%
gaus = Gaus(1,2)
sup = gaus.getSupport(10.0)

wavelet = np.zeros(sup*2+1,dtype='csingle')
gaus.getWavelet(10.0,wavelet)

waveletFT = np.zeros(sup*2+1,dtype='csingle')
gaus.getWaveletFT(10.0,waveletFT)

#%% Generate data
fs = 50     #Sampling frequency
n = 1024
x = np.arange(n)
y = np.sin(2*np.pi*x/20, dtype='float32')+np.sin(2*np.pi*x/50, dtype='float32')
hz = 5

f0 = 0.01   #Min frequency
f1 = 25    #Max frequency
fn = 50     #Number of frequencies

#%%
out = np.zeros((fn,len(x)), dtype='csingle')
freqs = np.zeros((fn), dtype='single')
scales = Scales(gaus,FCWT_LOGSCALES,fs,f0,f1,fn)

#In terms of how the log scale is created internally 
import math
#Smallest scale
s0 = fs/f1
#Largest scale
s1 = fs/f0
#Our scales are all 2^(something)
#So 2^log2(s0) = s0
#and 2^log2(s1) = s1
#In between we will make an even scale for the logs
#Each scale is 2^[(log2(s1)-log2(s0))/(N-1) * i]
#Here is our scales computation
delta_p = (math.log2(s1)-math.log2(s0))/(fn-1)
scales_test = np.array([2**(math.log2(s0) + delta_p*i) for i in range(fn) ])

#Here is internal to fCWT
scales_np = np.zeros((fn), dtype='single')
scales.getScales(scales_np)

#Lets compare
print('Our scales vs fCWT scales:')
print(np.linalg.norm(scales_np-scales_test))


#%%
scales.getFrequencies(freqs)
fcwt = FCWT(gaus, fs, False, True)

import time
start = time.time()
fcwt.cwt(y, scales, out)
end = time.time()
print(end - start)

#%% Reconstruct back 
# The reconstruction is explained in CWTReconstructionFactors.pdf 
# Our spacing between scales is delta_p
# Our time difference is 1/fs
dt = 1/fs

reconstruction = ( np.sqrt(dt) * delta_p * 
                  np.transpose(np.sum(np.transpose(np.real(out)), axis=-1)).transpose())
reconstruction = reconstruction * (1 / ( 3.5987 * 0.948 *4.5))

plt.figure(1)
plt.clf()
plt.plot(y,color='b')
plt.plot(reconstruction*100,color = 'r')

#%% Now apply cwt using a different library 
import pywt
import numpy as np
import matplotlib.pyplot as plt


pywt_scales = np.array(pywt.frequency2scale('gaus1', np.double(freqs)))
pywt_scales = pywt_scales[np.where(pywt_scales > 0.1)[0]]
coef, pywt_freqs=pywt.cwt(y,pywt_scales,'gaus1')
# using the variable axs for multiple Axes
fig, axs = plt.subplots(2, 1)

axs[0].matshow(np.real(out))
axs[1].matshow(np.real(coef))

#%%
import matplotlib.pyplot as plt
import numpy as np

import pywt

# plot complex morlet wavelets with different center frequencies and bandwidths
wavelets = [f"gaus{x:d}" for x in [2, 4, 6, 8]]
fig, axs = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)
for ax, wavelet_name in zip(axs.flatten(), wavelets):
    [psi, x] = pywt.ContinuousWavelet(wavelet_name).wavefun(10)
    ax.plot(x, np.real(psi), label="real")
    ax.plot(x, np.imag(psi), label="imag")
    ax.set_title(wavelet_name)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-0.8, 1])
ax.legend()
plt.suptitle("Complex Morlet Wavelets with different center frequencies and bandwidths")
plt.show()