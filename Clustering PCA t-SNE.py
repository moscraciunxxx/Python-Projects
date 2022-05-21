# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 02:09:43 2022

@author: moscr
"""
import numpy as np
import matplotlib.pyplot as plt 
import requests
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import time 

# Import and normaliza the cloud data 

CLOUD=requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/taylor/cloud.data')
cloud_data=CLOUD.text
#%% 
cloud=[]    
for aline in cloud_data:
    aline=aline.strip().split()
    try:
        l=list(map(float,aline))
        if  l:
            cloud.append(l)
    except: None

cloud=np.array(cloud)
cloudz=np.zeros(cloud.shape)
for coli in range(cloud.shape[1]):
    mean=np.mean(cloud[:,coli])
    std=np.std(cloud[:,coli],ddof=1)  # stats unbiased 
    cloudz[:,coli]=(cloud[:,coli]-mean)/std
    
fig,ax=plt.subplots(1,2,figsize=(15,10))
ax[0].plot(cloud)
ax[0].set_title('Raw data')

ax[1].plot(cloudz)
ax[1].set_title('Normalized data')


plt.show()


#%% compute and inspect covariance matrices 

# mean center  the data
cloud_demean=cloud-cloud.mean(axis=0)
plt.plot(cloud_demean)

cov_features=cloud_demean.T @ cloud_data/cloud_demean.shape[1]
cov_featuresZ=cloudz.T @ cloudz/(cloudz.shape[1]-1)

# covariance matrized for observations 


cov_observations=cloud_demean @ cloud_data.T/(cloud_demean.shape[0]-1)
cov_observationsZ=cloudz @ cloudz.T/(cloudz.shape[0]-1)



fig,ax=plt.subplots(2,2,figsize=(15,10))
ax[0,0].imshow(cov_features,vmin=-8e5,vmax=8e5)
ax[0,0].set_title('Raw features')
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])


ax[0,1].imshow(cov_featuresZ)
ax[0,1].set_title('Znorm features')
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])

ax[1,0].imshow(cov_observations)
ax[1,0].set_title('Raw observations')
ax[1,0].set_xticks([])
ax[1,0].set_yticks([])

ax[1,1].imwsho(cov_observationsZ)
ax[1,1].set_title('Znorm observations')
ax[1,1].set_xticks([])
ax[1,1].set_yticks([])

#%% Determine the number of components using PCA 

pca=PCA().fit(cloudz)
fig,ax=plt.subplots(1,2,figsize=(15,10))

ax[0].imshow(pca.get_covariance())
ax[1].plt(100*pca.explained_variance_ratio_,'ks',markersize=10,markerfacecolor='w')

plt.show()


# project the data down to 2 PCs

cloud2D=pca.transform(cloudz)
plt.plot(cloud2D[:,0],cloud2D[:,1],'o')
plt.xlabel('PC 1')
plt.ylabel('PC 2')

plt.show()


#%% Cluster the data using t-SNE and k-means

t=TSNE().fit_transform(cloudz)

# run k-means
clusters=KMeans(n_clusters=2).fit(t)

plt.plot(t[clusters.labels_==0,0],t[clusters.label_==0,1],'ro',markerfacecolor='w')
plt.plot(t[clusters.labels_==1,0],t[clusters.label_==1,1],'bo',markerfacecolor='w')

plt.show()

#%% 
perps=[10,50,100,200]
fig,ax=plt.subplots(1,4,figsize=(15,10))



for i in range(4):
    # start a timer 
    starttime=time.clock()
    
    
    t=TSNE(perplexity=perps[i]).fit_transform(cloudz)
    clusters=KMeans(n_clusters=2).fit(t)
    # stop the timer 
    endtime=time.clock()-starttime
    
    
    ax[i].plot(t[clusters.labels_==0,0],t[clusters.label_==0,1],'ro',markerfacecolor='w')
    ax[i].plot(t[clusters.labels_==1,0],t[clusters.label_==1,1],'bo',markerfacecolor='w')
    ax[i].set_title(f'Perplexity ={perps[i]},took {endtime:.3f} seconds')


#%% Bonus video  MAKE a 2D likelihood density plot 

# re-run the decomposition 
t=TSNE(perplexity=10).fit_transform(cloudz)
# setup the image 
trange=np.arange(-100,100)

timage=np.zeros((len(trange),len(trange)))

# map the sparse coordinates onto the dense matrix
for i in range(len(t)):
    xi=np.argmin((t[i,0]-trange)**2)
    yi=np.argmin((t[i,1]-trange)**2)
    timage[yi,xi]+=1
    
  
plt.plot(t[:,0],t[:,1],'ro',markerfacecolor='w')
plt.show()
plt.imshow(timage,vmin=0,vmax=1,origin="top",extent=[trange[0],trange[-1],trange[0],trange[-1]])
plt.show()


from scipy.ndimage import gaussian_filter
# smooth the image
timage_smooth=gaussian_filter(timage, sigma=(3,3))
plt.imshow(timage_smooth,vmin=0,vmax=.3,origin="top",extent=[trange[0],trange[-1],trange[0],trange[-1]])
plt.show()

# the idea of convolving a series of data points with a gaussian is that it captures the uncertainty around our estimate of each data point






























































