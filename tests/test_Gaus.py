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
gaus = Gaus(1.0,1)
sup = gaus.getSupport(10.0)

wavelet = np.zeros(sup*2+1,dtype='csingle')
gaus.getWavelet(10.0,wavelet)

#%% Generate data
fs = 1
n = 1024
x = np.arange(n)
y = np.sin(2*np.pi*x/20, dtype='float32')
hz = 5

f0 = 0.01
f1 = 100
fn = 50

#%%s
#gaus = Gaus(2.0,2)
#%%
out = np.zeros((fn,len(x)), dtype='csingle')
freqs = np.zeros((fn), dtype='single')
scales = Scales(gaus,FCWT_LOGSCALES,1,f0,f1,fn)

#%%
scales.getFrequencies(freqs)
fcwt = FCWT(gaus, 1, False, False)

import time
start = time.time()
fcwt.cwt(y, scales, out)
end = time.time()
print(end - start)


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