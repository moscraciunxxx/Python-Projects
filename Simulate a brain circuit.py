# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 10:14:56 2022

@author: moscr
"""


import matplotlib.pyplot as plt 
import numpy as np




#%% Simulate one brain cell 

# parameters 
a=0.02
b=.2
c=-50
d=2

v=-65
u=b*v


# loop over simulation time
simulation_time=1000 

membrane_volt=np.zeros(simulation_time)
I_all=np.zeros(simulation_time)


for t in range(simulation_time):
    
    # define the input strength
    
    I=-2  if (t>200) & (t<400) else 7
    
    
    # check if there is an action potential 
    
    if v>=30:
        v=c
        u+=d
        
    # update the membrane variables 
    
    v+= 0.04*v**2 + 5*v + 140 - u + I
    u+= a*(b*v-u)
    
    # collect the variables for subsequent plotting 
    membrane_volt[t]=v
    I_all[t]=I


#%%


# plotting 
fig,ax=plt.subplots(1,figsize=(25,8))
plt.plot(membrane_volt,'k',label="Membrane potential ")
plt.plot(I_all-90,'m',label='Stimulation ')
plt.box(False) # preserve the tickmarks but it removes the axis limits 
plt.legend(fontsize=15)

plt.show()



#%% Create a circuit of 1000 neurons 

# Excitatory cells                                      # Inhibitory cells 

Ne=800;                                                 Ni=200;
re=np.random.rand(Ne)**2;                               ri=np.random.rand(Ni)**2

a=np.hstack((.02*np.ones(Ne),                         0.02+0.08*ri))
b=np.hstack((.2*np.ones(Ne),                          0.25-0.05*ri))
c=np.hstack((-65+15*re,                             -65*np.ones(Ni)))
d=np.hstack((8-6*re,                                2*np.ones(Ni)))


v=-65*np.ones(Ne+Ni)
u=b*v


# S matrix - connectivity 
S=np.hstack((.5*np.random.rand(Ne+Ni,Ne), -np.random.rand(Ne+Ni,Ni)))

plt.imshow(S)

#%% Simulate the brain circuit 
simulation_time=5000 # 5 seconds 
firings=np.array([[],[]])

for t in range(simulation_time):
    
    # define the exogenous input 
    I=np.hstack((5*np.random.randn(Ne),2*np.random.randn(Ni)))
    
    # check for AP
    fired=np.where(v>=30)[0]   # gives us a list of the neurons where they emitted an AP 
    
    
    # store the spike indices and times 
        
    temp=np.stack((np.tile(t,len(fired)),fired))
    firings=np.concatenate((firings,temp),axis=1)
    
    
    # update membrane variables for neurons that spiked 
    v[fired]=c[fired]
    u[fired]+=d[fired]
    
    
    
    # update the I to include spiking activity 
    
    I+= np.sum(S[:,fired],axis=1)
    
    # update the membrane potential for all the neurons 
    v+= 0.04*v**2 + 5*v + 140 - u + I
    u+= a*(b*v-u)

#%% 


# plotting 
fig,ax=plt.subplots(1,figsize=(14,8))
plt.plot(firings[0,:],firings[1,:],'k.',markersize=1)
plt.show()

# Visualize population activity 
popact=np.zeros(simulation_time)

for t in range(simulation_time):
    popact[t]=100*np.sum(firings[0,:]==t)/(Ne+Ni)

# spectram analysis 

popactX=np.abs(np.fft.fft(popact-np.mean(popact)))**2  # mean center the zeo hz components  
hz=np.linspace(0,1000/2,int(simulation_time/2+1))


fig,ax=plt.subplots(1,2,figsize=(20,8))

ax[0].plot(popact)
ax[0].set_xlabel('Time (ms)')
ax[0].set_ylabel('Percentage of neurons active')
ax[0].set_title('Time domain')

ax[1].plot(hz,popactX[:len(hz)])
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Spectral power')
ax[1].set_title('Frequency domain')
ax[1].set_xlim([0,60])  # the DC offset 

#%% Run some experiments 

def simCircuit(I):

        
    firings=np.array([[],[]])
        
    v=-65*np.ones(Ne+Ni)
    u=b*v
    
    for t in range(len(I)):
        
        # define the exogenous input 
        stim=np.hstack((3*np.random.randn(Ne),1.1*np.random.randn(Ni)))
        stim+=I[t]
        
        # check for AP
        fired=np.where(v>=30)[0]   # gives us a list of the neurons where they emitted an AP 
        
        
        # store the spike indices and times 
            
        temp=np.stack((np.tile(t,len(fired)),fired))
        firings=np.concatenate((firings,temp),axis=1)
        
        
        # update membrane variables for neurons that spiked 
        v[fired]=c[fired]
        u[fired]+=d[fired]
        
        
        
        # update the I to include spiking activity 
        
        stim+= np.sum(S[:,fired],axis=1)
        
        # update the membrane potential for all the neurons 
        v+= 0.04*v**2 + 5*v + 140 - u + stim
        u+= a*(b*v-u)
        
        
    return firings 

def plotTopActivity(firings):
        
    npoints=int(np.max(firings[0,:])+1)   
    # Visualize population activity 
    popact=np.zeros(npoints)
    
    for t in range(npoints):
        popact[t]=100*np.sum(firings[0,:]==t)/(Ne+Ni)
        
        
    # spectram analysis 
    
    popactX=np.abs(np.fft.fft(popact-np.mean(popact)))**2  # mean center the zeo hz components  
    hz=np.linspace(0,1000/2,int(npoints/2+1))
            
    fig,ax=plt.subplots(1,3,figsize=(30,8))
    
    ax[0].plot(firings[0,:],firings[1,:],'k.',markersize=2)
    ax[0].set_title('All neuron firings')
    ax[0].plot(50*I+100,'m',linewidth=3)
    
    
    
    ax[1].plot(popact)
    ax[1].set_xlabel('Time (ms)')
    ax[1].set_ylabel('Percentage of neurons active')
    ax[1].set_title('Time domain')
    
    ax[2].plot(hz,popactX[:len(hz)])
    ax[2].set_xlabel('Frequency (Hz)')
    ax[2].set_ylabel('Spectral power')
    ax[2].set_title('Frequency domain')
    ax[2].set_xlim([0,60])  # the DC offset 
    plt.show()   
        
#%%
# experiment 1

I=np.ones(1000)
I[400:601]=-3


# Experiment 2 
I=np.linspace(-2,2,3000)**2

# Experiment 3 

I=np.sin(np.linspace(0,6*np.pi,2355))
# run the simulation and visualize the results 
networkSpikes=simCircuit(I)


plotTopActivity(networkSpikes)


#%% Bonus video: Separate excitation from inhibition 



def plotTopActivity_EI(firings):
        
    npoints=int(np.max(firings[0,:])+1)   
    # Visualize population activity 
    popact=np.zeros((2,npoints))
    
    for t in range(npoints):
        popact[0,t]=100*np.sum(firings[0,firings[1,:]<Ne]==t)/Ne  # excitatory population response 
        popact[1,t]=100*np.sum(firings[0,firings[1,:]>(Ne-1)]==t)/Ni  # inhibitory population response 
        
        
    # spectram analysis 
    
    
    popactXE=np.abs(np.fft.fft(popact[0,:]-np.mean(popact[0,:])))**2  # mean center the zeo hz components  
    popactXI=np.abs(np.fft.fft(popact[1,:]-np.mean(popact[1,:])))**2 
    hz=np.linspace(0,1000/2,int(npoints/2+1))
            
    fig,ax=plt.subplots(1,3,figsize=(30,8))
    
    ax[0].plot(firings[0,firings[1,:]<Ne],firings[1,firings[1,:]<Ne],'g.',markersize=2)
    ax[0].plot(firings[0,firings[1,:]>=Ne],firings[1,firings[1,:]>=Ne],'r.',markersize=2)
   
    ax[0].set_title('All neuron firings')
    ax[0].plot(50*I+100,'m',linewidth=3)
    
    
    
   
    ax[1].plot(popact[1,:],'r',label='I cells')
    ax[1].plot(popact[0,:],'g',label='E cells')
    ax[1].set_xlabel('Time (ms)')
    ax[1].set_ylabel('Percentage of neurons active')
    ax[1].set_title('Time domain')
    ax[1].legend(fontsize=15)
    
    ax[2].plot(hz,popactXE[:len(hz)],'g')
    ax[2].plot(hz,popactXI[:len(hz)],'r')
    ax[2].set_xlabel('Frequency (Hz)')
    ax[2].set_ylabel('Spectral power')
    ax[2].set_title('Frequency domain')
    ax[2].set_xlim([0,60])  # the DC offset 
    plt.show()   



I=np.sin(np.linspace(0,6*np.pi,2355))

I=np.random.randn(1400)+2
# run the simulation and visualize the results 
networkSpikes=simCircuit(I)
plotTopActivity_EI(networkSpikes)


























































