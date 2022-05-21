# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 10:06:48 2022

@author: moscr
"""

# Time frequencies plots are the same as spectrogram 

# Real-valued Morlet wavelets 

import numpy as np
import matplotlib.pyplot as plt 

# define functions 

def createRealWavelet(time,freq,fwhm):
    sinepart=np.cos(2*np.pi*freq*time)
    gausspart=np.exp((-4*np.log(2)*time**2)/(fwhm**2))
    
    return sinepart*gausspart

# parameters 
freq=5
fwhm=.5
srate=500
time=np.arange(-2*srate,2*srate+1/srate)/srate
npoints=len(time)
#%% 
# create the wavelet 

wavelet=createRealWavelet(time, freq, fwhm)

# get the power spectrum of the wavelet 

waveletX=np.abs(np.fft.fft(wavelet/npoints))**2
hz=np.linspace(0,srate/2+1,int(npoints/2+1))


# create a figure with a 1x2 subplot geometry 

fig,ax=plt.subplots(1,2,figsize=(20,10))


ax[0].plot(time,wavelet);
ax[0].set_xlabel('Time (s)')
ax[0].set_title('Time domain')

ax[1].stem(hz,waveletX[:len(hz)],'k',use_line_collection=True)
ax[1].set_xlim([0,20])
ax[1].set_xlabel('Freequency (hz)')
ax[0].set_title('Frequency domain')


#%% Complex-valued Morlet wavelets 


def createComplexWavelet(time,freq,fwhm):
    sinepart=np.exp(1j*2*np.pi*freq*time)
    gausspart=np.exp((-4*np.log(2)*time**2)/(fwhm**2))
    
    return sinepart*gausspart



wavelet=createComplexWavelet(time, freq, fwhm)  
plt.plot(time,np.real(wavelet),label='Real part')
plt.plot(time,np.imag(wavelet),label='Imaginary part')
plt.plot(time,np.abs(wavelet),'k',label='Magnitude part')
plt.legend()
plt.xlabel('Time (s)')
plt.show()




plt.plot(time,np.abs(wavelet),'k',label='Magnitude part')
plt.plot(time,np.angle(wavelet),'m',label='Phase angle')
plt.ylabel('Angle (rad) or amplitude in (a.u.)')
plt.legend()
plt.show()


#%% Create a wavelet family 

# define the parameters 

lowfreq=2
highfreq=80
numfreqx=45

freqx=np.linspace(lowfreq,highfreq,numfreqx)
fwhm=np.linspace(5,1,numfreqx)

waveletfam=np.zeros((numfreqx,npoints),dtype=complex)


for wi in range(numfreqx):
    waveletfam[wi,:]=createComplexWavelet(time, freqx[wi], fwhm[wi])

plt.plot(time,np.real(waveletfam[5,:]))
plt.title(str(freqx[5]))


plt.plot(time,np.abs(waveletfam[5,:]))


#%%
fig,ax=plt.subplots(1,3,figsize=(20,7))

# show the real part of the wavelet family 
ax[0].imshow(np.real(waveletfam),aspect='auto',origin='lower',extent=[time[0],time[-1],lowfreq,highfreq],
                                                                      vmin=-.7,vmax=.7 )
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Frequency  (hz)')
ax[0].set_title('The real part')


# show the phase angles of the wavelet family 
ax[1].imshow(np.angle(waveletfam),aspect='auto',origin='lower',extent=[time[0],time[-1],lowfreq,highfreq],
                                                                      vmin=-3.1,vmax=3.1 )
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Frequency  (hz)')
ax[1].set_title('The phase angles ')



# show the magnitude of the wavelet family 
ax[2].imshow(np.abs(waveletfam),aspect='auto',origin='lower',extent=[time[0],time[-1],lowfreq,highfreq],
                                                                      vmin=-.01,vmax=.7 )
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('Frequency  (hz)')
ax[2].set_title('The magnitude')


plt.show()

#%% Import and visualize the EEG data 
from scipy.io import loadmat 
EEG=loadmat('sampleEEGdata.mat')

# extract the necessary information 
times=np.squeeze(EEG['EEG'][0][0][14])
data=EEG['EEG'][0][0][15]
fs=EEG['EEG'][0][0][11][0][0].astype(int)  # the sampling rate 


#%%create the ERP  

erp=np.mean(data[46,:,:],axis=1) # because we are slicing out and the data is no longer 3D but 2D

plt.plot(times,erp)
plt.xlim([-200,1000])
plt.xlabel('Time (ms)')
plt.ylabel('Voltage ($\mu V$)')
plt.title('ERP from channle 47')
plt.show()

#%% Wavelet convolution 
# define the parameters 

lowfreq=2
highfreq=30
numfreqx=45
time=np.arange(-fs,fs+1)/fs

freqx=np.linspace(lowfreq,highfreq,numfreqx)
fwhm=np.linspace(1.5,.5,numfreqx)

waveletfam=np.zeros((numfreqx,len(time)),dtype=complex)

for wi in range(numfreqx):
    waveletfam[wi,:]=createComplexWavelet(time, freqx[wi], fwhm[wi])

# plot some selected wavelets 

for i in range(5):
    plt.plot(time,np.real(waveletfam[i*4,:])+i*1.5 )

plt.xlabel('Time (ms)')
plt.tick_params(labelleft=False,labelbottom=False)

plt.show()
#%% Convolution 

convres=np.convolve(erp,waveletfam[0,:],mode='same')

plt.plot(times,np.real(convres),label='Real part')
plt.plot(times,np.abs(convres),label='The Magnitude')
plt.plot([times[0],times[-1]],[0,0],'k--')
ylim=plt.ylim()
plt.plot([0,0],ylim,'k:')
plt.xlim([times[0],times[-1]])
plt.ylim(ylim)

plt.legend()
plt.show()

#%% Create a time-frequency map 

tf=np.zeros((numfreqx,len(times)))

for wi in range(numfreqx):
    convres=np.convolve(erp,waveletfam[wi,:],mode='same')
    tf[wi,:]=np.abs(convres)
    
    
plt.imshow(tf,origin='lower',aspect='auto',vmax=130,extent=[times[0],times[-1],lowfreq,highfreq])

plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')

plt.show()   



#%% Bonus : Phase map with circular colormap 





phases=np.zeros((numfreqx,len(times)))

for wi in range(numfreqx):
    convres=np.convolve(erp,waveletfam[wi,:],mode='same')
    phases[wi,:]=np.angle(convres)
    
    
plt.imshow(phases,origin='lower',aspect='auto',vmin=-3,vmax=3,extent=[times[0],times[-1],lowfreq,highfreq],cmap='hsv')


plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.colorbar()

plt.show()   







































































