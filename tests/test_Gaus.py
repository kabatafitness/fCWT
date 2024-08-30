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

%matplotlib qt

#%%
gaus = Gaus(1,2)
sup = gaus.getSupport(10.0)

wavelet = np.zeros(sup*2+1,dtype='csingle')
gaus.getWavelet(10.0,wavelet)

waveletFT = np.zeros(sup*2+1,dtype='csingle')
gaus.getWaveletFT(10.0,waveletFT)

#%% Generate data
fs = 50     #Sampling frequency
n = 2024
x = np.arange(n)/100

# The signal will be random sum of sinusoids frequencies from 0.5Hz to 5Hz
# We will also add a bias term 

freqs = np.linspace(0.5, 2, 100)*2*np.pi #Frequencies in radians
f_sigma = 1;
amps = np.random.randn(freqs.size)*f_sigma

signal = np.zeros(n)
for i in range(freqs.size):
    signal = signal + amps[i]*np.sin(x*freqs[i])

bias = 2
slope = 0.5
signal = signal + bias + x*slope

y = np.array(signal, dtype='float32')
hz = 5

f0 = 0.5   #Min frequency
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
                  np.sum(np.transpose(np.real(out)), axis=-1))
reconstruction = reconstruction * 20.5178 * (1 / ( 3.5987 * 0.867))

#reconstruction2 = np.sqrt(dt) * delta_p *np.sqrt(np.pi*2*scales_np/dt) @np.real(out) * (1 / ( 3.5987 * 0.867))


plt.figure(1)
plt.clf()
plt.plot(y,color='b')
plt.plot(reconstruction,color = 'r')
plt.plot(y-bias-x*slope,color='g')