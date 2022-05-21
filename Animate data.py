# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 06:21:49 2022

@author: moscr
"""
import numpy as np
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
import matplotlib.animation as animation
from matplotlib import rc
rc('animation',html='jshtml')


# %matplotlib qt

#%% Wavey wavelets in plotly 



# function for complex wavelet 

def createComplexWavelet(time,freq,fwhm,phs=0):
    sinepart=np.exp(1j*(2*np.pi*freq*time +phs))
    gausspart=np.exp((-4*np.log(2)*time**2)/(fwhm**2))
    
    return sinepart*gausspart


#%% create and visualize one wavelet 

freq=5
fwhm=.5
srate=500 #Hz
time=np.arange(-2*srate,2*srate)/srate

wavelet=createComplexWavelet(time, freq, fwhm,phs=np.pi)

plt.plot(time,np.real(wavelet),label='Real part')
plt.plot(time,np.imag(wavelet),label='Imaginary part ')
plt.plot(time,np.abs(wavelet),label='Magnitude ')
plt.legend(frameon=False)  # gets rid of the box encircling the legend text 
plt.show()


#%% 
fig=go.Figure(
    data=[go.Scatter(x=time,y=np.real(createComplexWavelet(time, freq, fwhm,0)),mode='lines')],
    
    layout=go.Layout(updatemenus=[dict(type='buttons',buttons=[{'label':'Play','method':'animate','args':[None]}])]),
    
    frames=[ go.Frame( data=[go.Scatter(x=time,y=np.real(createComplexWavelet(time, freq, fwhm,np.pi/6)),mode='lines')]),
            go.Frame(data=[go.Scatter(x=time,y=np.real(createComplexWavelet(time, freq, fwhm,np.pi/4)),mode='lines')]),
            go.Frame(data=[go.Scatter(x=time,y=np.real(createComplexWavelet(time, freq, fwhm,np.pi/2)),mode='lines')])]
    
    
    )

fig.show()


#%% 
gofigure={
    
    'data':[go.Scatter(x=time,y=np.real(createComplexWavelet(time, freq, fwhm,0)),mode='lines',name='Real part'),
            go.Scatter(x=time,y=np.imag(createComplexWavelet(time, freq, fwhm,0)),mode='lines',name='Imaginary part')],
        
        
    'layout':go.Layout(updatemenus=[dict(type='buttons',
            buttons=[{'label':'Play','method':'animate','args':[None]}])],title='Complex Morlet wavelet'),
        
        
     'frames':[]   
    
    
    
    
    }

#%% create frames
 

phases=np.linspace(0,2*np.pi,20)

for theta in phases:
    frames={'data':[]}
    tempdict1={'x':time,'y': np.real(createComplexWavelet(time, freq, fwhm,theta))} 
    tempdict2={'x':time,'y': np.imag(createComplexWavelet(time, freq, fwhm,theta))}
    frames['data'].append(tempdict1) 
    frames['data'].append(tempdict2)
    gofigure['frames'].append(frames)


figure=go.Figure(gofigure)


figure.show()

#%% Wavey wavelets in matplotlib  

def aframe(phases):
    
    # create a new wavelet 
    
    wavelet=createComplexWavelet(time, freq, fwhm,phases)
    
    # update the plot 
    plth1.set_ydata(np.real(wavelet))
    plth2.set_ydata(np.imag(wavelet))
    
    return (plth1,plth2)

# setup a figure 
fig,ax=plt.subplots(1,figsize=(15,8))

plth1,=ax.plot(time,np.zeros(len(time)))  # tells python we need only the first output 
plth2,=ax.plot(time,np.zeros(len(time)))
ax.set_ylim([-1.1,1.1])

# create an animation object 


ani=animation.FuncAnimation(fig, aframe, phases,interval=75,repeat=True)


#%% Mobius transform in matplotlib 

a=np.linspace(-40,40,90)
b=np.linspace(-40,40,80)

mobtrans=np.zeros((len(a),len(b)),dtype=complex)
ts=np.linspace(.2,2,99)

def afram(t):
    for i,aa in enumerate(a):
        for j,bb in enumerate(b):       
            z=np.complex(aa,bb)   
            #create the mobius image 
            num=(t-1)+(t+1)*z
            denom=(t+1)+(t-1)*z
            mobtrans[i,j]=num/denom
            
            
    # update the frame 
    imh[0].set_data(np.real(mobtrans))
    imh[1].set_data(np.imag(mobtrans))
    imh[2].set_data(np.abs(mobtrans))
    return imh
    
fig,axs=plt.subplots(1,3,figsize=(20,12))
imh=[0]*3

imh[0]=axs[0].imshow(np.zeros((len(a),len(b))),vmin=-50,vmax=50)
imh[1]=axs[1].imshow(np.zeros((len(a),len(b))),vmin=-50,vmax=50)
imh[2]=axs[2].imshow(np.zeros((len(a),len(b))),vmin=-50,vmax=50)
for i in range(3):
    axs[i].tick_params(labelbottom=False,labelleft=False)
axs[0].set_title('Real part')
axs[1].set_title('Imaginary part')
axs[2].set_title('Magnitude')
        
ani=animation.FuncAnimation(fig, aframe, ts,interval=50)


# save as gif 
writergif=animation.PillowWriter(fps=30)
ani.save('Mobius.gif',writer=writergif)  


#%% Bonus video The wandering prime 

from sympy import isprime
n=int(1e6)
primes=[i for i in range(n) if isprime(i)]
direction=0

# initialize the matrix 

xy=np.zeros((len(primes),2))

for i in range(len(primes)):
    dist =primes[i]-primes[i-1]
    if direction==0:
    # update the xy coordinates 
        xy[i,:]=[xy[i-1,0],xy[i-1,1] +dist]
    elif direction==1:
        xy[i,:]=[xy[i-1,0]+dist,xy[i-1,1] ]
    elif direction ==2:
        xy[i,:]=[xy[i-1,0],xy[i-1,1] -dist] 
    elif direction ==3:
        xy[i,:]=[xy[i-1,0]-dist,xy[i-1,1] ]
    
    
    # update the direction  direction=(direction+1)%4
    # direction+=1
    # if direction ==4:
    #     direction=0
    direction=(direction+1)%4
# mean center the coordinates

xy-=np.mean(xy,axis=0)
     

plt.plot(xy[:,0],xy[:,1],'k')
plt.show()
        

#%% function for drawing each frame 

def aframe(i):
    ph.set_xdata(xy[:i,0])
    ph.set_ydata(xy[:i,1])
    
    return ph
    




# setup the image 
fig,ax=plt.subplots(1,figsize=(15,9))
ph,=ax.plot(xy[:,0],xy[:,1],'k')
ax.set_aspect(1/ax.get_data_ratio())
ax.axis('off')


# create the animation
skip_step=100
ani=animation.FuncAnimation(fig, aframe, range(0,len(xy),skip_step))



    





































































































































