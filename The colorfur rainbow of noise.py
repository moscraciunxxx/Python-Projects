# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 04:12:46 2022

@author: moscr
"""

import numpy as np
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt 

npoints=int(1e5)

whitenoise=np.random.randn(npoints)  # randn - the mean is zero (of the noise)

brownnoise=np.cumsum(whitenoise)

#%% 
plt.plot(whitenoise, color=[1,1,0],label="White noise ")
plt.plot(brownnoise,color=[.4,.2,.07], label="Brown noise ")
plt.legend()
plt.show()

#%% Plot YY axis 

fig, ax=plt.subplots(1)

ax.plot(whitenoise,color=[1,1,0],label="White noise ")
ax.set_ylabel("White noise ")
plt.legend()
ax2=ax.twinx()
ax2.plot(brownnoise,color=[.4,.2,.07],label="Brown noise ")
ax2.set_ylabel("Brown noise ")
plt.legend()


plt.show()


#%% Power spectrum 

whitenoiseX=np.abs(fftpack.fft(whitenoise))/npoints  # normalization factor 
brownnoiseX=np.abs(fftpack.fft(brownnoise))/npoints

# vector of frequencies 

frex=np.linspace(0,1,int(npoints/2+1))


plt.plot(frex,brownnoiseX[:len(frex)],'.',color=[.4,.2,.07])
plt.plot(frex,whitenoiseX[:len(frex)],'.',color=[1,1,0])
plt.ylim([0,0.03])
plt.xlabel('Frequency (fraction of Nyquist)')
plt.ylabel("Amplitude (a.u)")
plt.show()



#%% Pink and blue noise 

fc_amp=1/(frex+ 0.01) +np.random.randn(int(npoints/2+1))**2*5

fc_phs = 2*np.pi*np.random.rand(int(npoints/2+1))    # uniformily distributed numbers between 0 and 1 
FourierSpectrum=np.zeros(npoints,dtype=complex)
FourierSpectrum[:int(npoints/2+1)] =fc_amp*np.exp(1j*fc_phs)

pinknoise=np.real(fftpack.ifft(FourierSpectrum))



plt.plot(pinknoise)
#%% 
pinknoiseX=np.abs(fftpack.fft(pinknoise))/npoints

fig,ax=plt.subplots(1,2,figsize=( 15,7))
ax[0].plot(pinknoise,color=[1,0,1])
ax[0].set_title('Pink noise in the time domain')

ax[1].plot(frex,pinknoiseX[:len(frex)],'.',color=[1,0,1])
ax[1].set_title('Pink noise in the frequency domain')


plt.show()


#%% Blue noise 

fc_amp=np.linspace(1,3,int(npoints/2+1))+np.random.randn(int(npoints/2+1))/5

fc_phs = 2*np.pi*np.random.rand(int(npoints/2+1))    # uniformily distributed numbers between 0 and 1 
FourierSpectrum=np.zeros(npoints,dtype=complex)
FourierSpectrum[:int(npoints/2+1)] =fc_amp*np.exp(1j*fc_phs)

bluenoise=np.real(fftpack.ifft(FourierSpectrum))

#%% Plotting 


bluenoiseX=np.abs(fftpack.fft(bluenoise))/npoints

fig,ax=plt.subplots(1,2,figsize=( 15,7))
ax[0].plot(bluenoise,color=[0,0,1])
ax[0].set_title('Blue noise in the time domain')

ax[1].plot(frex,bluenoiseX[:len(frex)],'.',color=[0,0,1])
ax[1].set_title('Blue noise in the frequency domain')


plt.show()

#%% The colorful specral rainbow 

def whiteNoiseSpect(amp):
    noise=amp*np.random.randn(npoints)
    return abs(fftpack.fft(noise)/ npoints)


def brownNoiseSpect(amp):
    noise=np.cumsum(amp*np.random.randn(npoints))
    return abs(fftpack.fft(noise)/ npoints)


def pinkNoiseSpect(amp):
    FourierSpectrum=np.zeros(npoints,dtype=complex)
    
    fc_amp=1/(frex+ 0.01) +np.random.randn(int(npoints/2+1))**2*5
    fc_phs = 2*np.pi*np.random.rand(int(npoints/2+1)) 
    FourierSpectrum[:int(npoints/2+1)] =fc_amp*np.exp(1j*fc_phs)
    noise=amp*np.real(fftpack.ifft(FourierSpectrum))
    return abs(fftpack.fft(noise)/ npoints)


def blueNoiseSpect(amp):
    
    fc_amp=np.linspace(1,3,int(npoints/2+1))+np.random.randn(int(npoints/2+1))/5
    fc_phs = 2*np.pi*np.random.rand(int(npoints/2+1))    # uniformily distributed numbers between 0 and 1 
    FourierSpectrum=np.zeros(npoints,dtype=complex)
    FourierSpectrum[:int(npoints/2+1)] =fc_amp*np.exp(1j*fc_phs)
    noise=amp*np.real(fftpack.ifft(FourierSpectrum))
    return abs(fftpack.fft(noise)/npoints)



plt.plot(frex,brownNoiseSpect(1)[:len(frex)],color=[.4,.2,0.07])
plt.plot(frex,whiteNoiseSpect(1)[:len(frex)],color=[1,1,0])
plt.plot(frex,pinkNoiseSpect(50)[:len(frex)],color=[1,0,1])
plt.plot(frex,blueNoiseSpect(1000)[:len(frex)],color=[0,0,1])
plt.ylim([0,0.03])
plt.title("Colorful frequency  domain plot of noise")
plt.xlabel("Frequency (fraction of Nyquist")
plt.ylabel("Ampltitude (a.u.)")


#%% Bonus video : how do they sound 

from IPython.display import Audio 

print('White noise ')
audio=Audio(whitenoise,rate=44100)


print('Brown noise ')
audio=Audio(brownnoise,rate=44100)



print('Pink noise ')
audio=audio=Audio(pinknoise,rate=44100)



print('Blue noise ')
Audio(bluenoise,rate=44100)
























































































