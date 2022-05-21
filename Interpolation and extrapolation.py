# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 11:40:42 2022

@author: moscr
"""


import numpy as np
import matplotlib.pyplot as plt 
from scipy import signal 
from scipy import interpolate

#%% Downsample and upsample a time series 

 

N=1500

timevec=np.arange(N)/N
tso=np.cumsum(np.random.randn(N))


# apply a mean-smoothing filter 

k=13
ts=np.convolve(tso,np.ones(k)/k,mode='same')  # we normalize 



plt.plot(timevec,tso)
plt.plot(timevec,ts)
plt.show()



# down-sample 
ts_ds=signal.resample(ts,int(N/2))   
timevec_ds=signal.resample(timevec,int(N/2))


plt.plot(timevec,ts,'r')
plt.plot(timevec_ds,ts_ds,'b.')
plt.show()

# up-sample
ts_us=signal.resample(ts,int(N*3))   
timevec_us=signal.resample(timevec,int(N*3))


plt.plot(timevec,ts,'r.',markersize=10)
plt.plot(timevec_us,ts_us,'b.')
plt.show()



#%% 1-D interpolation 

sig=np.array([1,3,-2,2,9,10,15,5,8])

plt.plot(sig,'o',markersize=15)
plt.show()



# interpolation using Numpy

interp_factor=3
interp_points=np.linspace(0,len(sig)-1,int(len(sig)*interp_factor))
origin_points=np.arange(len(sig))



interp_sig=np.interp(interp_points,origin_points,sig)

plt.plot(sig,'o',markersize=15)
plt.plot(interp_points,interp_sig,'rs-')
plt.show()

#%% using Scipy 


interp_funL=interpolate.interp1d(origin_points,sig,kind='linear')
interp_sigL=interp_funL(interp_points)



interp_funC=interpolate.interp1d(origin_points,sig,kind='cubic')
interp_sigC=interp_funC(interp_points)

plt.plot(origin_points,sig,'o',markersize=15,label="Original")
plt.plot(interp_points,interp_sigL,'rs-',label="Linear")
plt.plot(interp_points,interp_sigC,'ms-',label="Cubic")
plt.legend()
plt.show()


#%% 1-D extrapolation 


extrap_points=np.linspace(0,(len(sig)-1)*2,int(len(sig)*interp_factor))

interp_funL=interpolate.interp1d(origin_points,sig,kind='linear',
                                 bounds_error=False,fill_value='extrapolate')
interp_sigL=interp_funL(extrap_points)



interp_funC=interpolate.interp1d(origin_points,sig,kind='cubic',
                                 bounds_error=False,fill_value='extrapolate')
interp_sigC=interp_funC(extrap_points)

plt.plot(origin_points,sig,'o',markersize=15,label="Original")
plt.plot(extrap_points,interp_sigL,'rs-',label="Linear")
plt.plot(extrap_points,interp_sigC,'ms-',label="Cubic")
plt.ylim([-4,30])
plt.legend()
plt.show()


#%% Resampling revisited 



interp_funL=interpolate.interp1d(timevec_us,ts_us)  # ts_us is evaluated on timevec points 
timevec_usRegular=np.linspace(timevec[0],timevec[-1],len(timevec_us))

ts_usRegular=interp_funL(timevec_usRegular)




plt.plot(timevec,ts,'rs',label="Original")
plt.plot(timevec_us,ts_us+5,'b.',label="Upsampled")
plt.plot(timevec_usRegular,ts_usRegular+8,'k.',label="Upsampled/Interpolated")
plt.legend()
plt.xlim([.6,.603])
plt.show()  

#%% Fix corrupted image with interpolation 

x=np.linspace(-2*np.pi,2*np.pi, 120)
X,Y=np.meshgrid(x, x)

Q=np.log( (X-2)**2 +np.abs((Y-3)**3))

Zgood=np.sin(Q)

plt.imshow(Zgood,extent=[x[0],x[-1],x[0],x[-1]],origin='lower')
plt.show()

         

# select pixels to demolish
num_elements=np.prod(Zgood.shape)
prop_bad_pixels=.15
badpix_idx=np.random.rand(int(num_elements*prop_bad_pixels))*num_elements
badpix_idx=np.floor(badpix_idx).astype(int)

#%% Destroy the pixels 

# how do we convert to rows ans columns indices from liner indices from  ?

i,j=np.unravel_index(badpix_idx,Zgood.shape)

Z=Zgood.copy()


Z[i,j]=np.nan

# find a list with the good and bad pixels 
bad_idx_i,bad_idx_j=np.where(np.isnan(Z))
good_idx_i,good_idx_j=np.where(np.isfinite(Z))


# interpolation instance usign griddata 

Znew_pix=interpolate.griddata((good_idx_i,good_idx_j), Z[good_idx_i,good_idx_j], (bad_idx_i,bad_idx_j))


Zinterp=Z.copy()
Zinterp[bad_idx_i,bad_idx_j]=Znew_pix


# visualization

fig,ax=plt.subplots(1,3,figsize=(20,10))
ax[0].imshow(Zgood)
ax[0].set_title("Original")

ax[1].imshow(Z)
ax[1].set_title("Corrupted")

ax[2].imshow(Zinterp)
ax[2].set_title("Interpolated")

plt.show()



#%% Bonus video : Draw a Necker cube 

from matplotlib.patches import Rectangle
o=1/3
r1=Rectangle((0,0), 1, 1,edgecolor='k',linewidth=5 ,facecolor='none')
r2=Rectangle((o,o), 1, 1,edgecolor='k',linewidth=5 ,facecolor='none')



fig,ax=plt.subplots(figsize=(20,15))
ax.add_patch(r1)
ax.add_patch(r2)

plt.plot([0,o],[0,o],'k',linewidth=3)
plt.plot([0,o],[1,1+o],'k',linewidth=3)

plt.plot([1,1+o],[0,o],'k',linewidth=3)
plt.plot([1,1+o],[1,1+o],'k',linewidth=3)

# plt.plot(0,0,'ko',markerfacecolor='w',markersize=15)
# plt.plot(1,0,'ko',markerfacecolor='w',markersize=15)
# plt.plot(1,1,'ko',markerfacecolor='w',markersize=15)
# plt.plot(0,1,'ko',markerfacecolor='w',markersize=15)

# plt.plot(o,o,'ko',markerfacecolor='w',markersize=15)
# plt.plot(o,1+o,'ko',markerfacecolor='w',markersize=15)
# plt.plot(1+o,o,'ko',markerfacecolor='w',markersize=15)
# plt.plot(1+o,1+o,'ko',markerfacecolor='w',markersize=15)


edges=[[0,0],[0,1],[1,0],[1,1],[o,o],[o,1+o],[1+o,o],[1+o,1+o]]
for i in edges:
    plt.plot(i[0],i[1],'ko',markerfacecolor='w',markersize=15)




ax.set_xlim([-.5,1.5])
ax.set_ylim([-.5,1.5])
plt.axis('off')

plt.show()
















































































































