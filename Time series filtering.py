# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 09:20:09 2022

@author: moscr
"""

import numpy as np
import matplotlib.pyplot as plt 
import scipy.fftpack
from scipy import signal 
# generate a signal with random noise 
#%% 
srate=1100

time=np.arange(10*srate)/srate
npoints=len(time)

# create the signal 

data=np.random.randn(npoints)*5

# add 60 Hz line noise 
data+=np.sin(2*np.pi*60*time)
#%%

def plotSignal(data):
    # create a figure
    fig,ax=plt.subplots(1,2,figsize=(18,8))
    # plot the time data 
    ax[0].plot(time,data)
    ax[0].set_xlabel('Time (s)')
    ax[0].set_title('Time domain')
    
    # plot the frequency domain signal 
    dataX=np.abs(scipy.fftpack.fft(data/npoints)) # normalize by divinding by the number of time points 
    hz=np.linspace(0,srate/2,int(npoints/2)+1)
    ax[1].plot(hz,dataX[:len(hz)])
    ax[1].set_xlim([0,250])
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_title('Frequency domain')
    
    
plotSignal(data)


#%% Notch out line noise 

f0=60  # frequency to be notched out 
Q=60 # quality of the filter 
# desing our notch filter 

b,a=signal.iirnotch(f0, Q,srate)

# evaluate the filter 
freq,h=signal.freqz(b,a,fs=srate)
plt.plot(freq,np.abs(h)**2)
plt.title('Frequency response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.xlim([0,150])

plt.show()

#%% 
notched=signal.filtfilt(b,a,data)

plotSignal(data)
plotSignal(notched)

#%% 


def plot2Signals(data1,data2):
    # create a figure
    fig,ax=plt.subplots(1,2,figsize=(18,8))
    # plot the time data 
    ax[0].plot(time,data1,label="Original signal")
    ax[0].plot(time,data2,label="Filtered signal  ")
    ax[0].set_xlabel('Time (s)')
    ax[0].legend()
    ax[0].set_xlim([1,1.2])
    ax[0].set_title('Time domain')
    
    # plot the frequency domain signal 
    data1X=np.abs(scipy.fftpack.fft(data1/npoints)) # normalize by divinding by the number of time points 
    data2X=np.abs(scipy.fftpack.fft(data2/npoints))
    hz=np.linspace(0,srate/2,int(npoints/2)+1)
    ax[1].plot(hz,data1X[:len(hz)],label='Original')
    ax[1].plot(hz,data2X[:len(hz)],label='Filtered signal')
    ax[1].set_xlim([0,250])
    ax[1].legend()
    
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_title('Frequency domain')
    

plot2Signals(data,notched)



#%% High-pass FIR filter 

lowedge=30 # Hz 

fkern =signal.firwin(lowedge*10+1,lowedge,fs=srate,pass_zero=False)  
# fkern =signal.firwin(lowedge*10+1,lowedge/(srate/2),pass_zero=False) we are scalling this cut-off freq according to the Nyquist frequency 

# evaluate the filter 
freq,h=signal.freqz(fkern,1,fs=srate)
plt.plot(freq,np.abs(h)**2)
plt.title('Frequency response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.xlim([0,150])

highpass=signal.filtfilt(fkern,1,notched)

plotSignal(highpass)
plot2Signals(data,highpass)


#%% Lowpass IIR filter 

lowcutoff=40
N=6
b,a=signal.butter(N,lowcutoff/(srate/2))  # normalize to a Nyqist frequency of 1


# evaluate the filter 
freq,h=signal.freqz(b,a,fs=srate)
plt.plot(freq,np.abs(h)**2)
plt.title('Frequency response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.xlim([0,150])


bandpass=signal.filtfilt(b,a,highpass)
plot2Signals(data,bandpass)

#%% Bonus video Make a desert landscape 

centerfreq=np.linspace(8,20,15)
filtdata=np.zeros((15,npoints))
r=0

for i in centerfreq:
    
    # design the filter kernel
    fwin=[i-2,i+2]
    b=signal.firwin(2004,fwin,pass_zero=False,fs=srate)
    
    # filter the data (store the result in a matrix)
    filtdata[r,:]=signal.filtfilt(b,1,data)
    r+=1
    
    
plt.imshow(filtdata,extent=[time[0],time[-1],centerfreq[0],centerfreq[-1]],origin='upper',cmap='Wistia',
           aspect='auto')

 
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

plt.xlim([4,5])



























































































