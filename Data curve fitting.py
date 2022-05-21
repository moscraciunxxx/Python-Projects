# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 02:44:16 2022

@author: moscr
"""

#%% Fit a Gaussian 
#  ! everything after the exclamation point is unix code 
import numpy as np
import matplotlib.pyplot as plt 
import lmfit.models as models 
from lmfit import Model 
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')   # make the graphs sharper


#create some noisy data 
gaussmodel=models.gaussian
# define 4 variables 

x=np.linspace(-3,3,353)
a=9 # amp 
c=.5 # center 
s=.35  # shape 

data=gaussmodel(x,a,c,s) + np.random.randn(len(x))

plt.plot(data,'s')

plt.show()
 
#%% 
gaussFit=Model(gaussmodel)
result=gaussFit.fit(data,x=x)


# R2 fit 
R2=np.corrcoef(data,result.best_fit)[0,1]**2


# visualize 

plt.plot(x,data,'o',label='Data')
plt.plot(x,result.init_fit,'k--',label='Initial guess')
plt.plot(x,result.best_fit,'r',label='Best fit')
plt.title('Model $R^2$ =%.3f'%R2)
plt.legend()
plt.show()

#%% Fit an exponential decay 


x=np.linspace(1,10,200)


data=1/x**2+np.random.normal(loc=0,scale=np.sqrt(.01),size=len(x))  # variance of 0.01

plt.plot(x,data,'.')
plt.xlabel('x')
plt.ylabel('$y=f(x)$' )
plt.show()


expdecay_fit=Model(models.exponential)
result=expdecay_fit.fit(data,x=x)

result.plot()

#%% Fit a sigmoid function 

def sigmoid(x,alpha=1,beta=1,tau=0):  # the default values need to be setted for the model to be able to fit the data 
    eterm=np.exp(-beta*(x-tau))
    return a/(1+eterm)


x=np.linspace(-3,3,367)
y=sigmoid(x,alpha=2,beta=5,tau=.1)+np.random.randn(len(x))

plt.plot(x,y,'s')
plt.show()
              

sigmodel=Model(sigmoid)
result=sigmodel.fit(y,x=x)


#%% visualize 



# R2 fit 
R2=np.corrcoef(y,result.best_fit)[0,1]**2


plt.plot(x,y,'o',label='Data')
plt.plot(x,result.init_fit,'k--',label='Initial guess')
plt.plot(x,result.best_fit,'r',label='Best fit')
plt.title('Model $R^2$ =%.3f'%R2)
plt.legend()
plt.show()




#%% COnjunctive model fitting 

# create the data 
x=np.linspace(-15,30,2000)
y=np.sin(x)/x + np.linspace(-1,1,len(x))

plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')

plt.show()

#%% creat the conjuncting model 

conj_model=Model(models.gaussian)+Model(models.linear)
result=conj_model.fit(y,x=x)

result.plot()
plt.show()

result.plot_fit(ylabel='The Functions',xlabel='The x axis',show_init=False)


#%% Multivariate model fitting 

# create the data 

n=300
theta=np.linspace(0,np.pi*2,n)
r=3
# x and y values 

x=r*np.cos(theta) + np.random.normal(0,1/r,n)
y=r*np.sin(theta) +np.random.normal(0,1/r,n)

plt.plot(x,y,'k.')
plt.axis('off')
plt.gca().set_aspect(1/plt.gca().get_data_ratio()) # access the axis 


plt.show()


# print(np.var(np.random.normal(0,1/r,n*100)))
# print(r**(-2))

#%%  
def circlefit(theta,r=1):
    x=r*np.cos(theta)
    y=r*np.sin(theta)
    return x,y 


result=Model(circlefit).fit((x,y),theta=theta)
fig= plt.figure(figsize=(10,8))
plt.plot(x,y,'b.',label="Data")
plt.plot(result.init_fit[0],result.init_fit[1],'k--',label='First approximation')
plt.plot(result.best_fit[0],result.best_fit[1],'r',label='Final approximation')
plt.legend(loc='upper left')
plt.gca().set_aspect(1/plt.gca().get_data_ratio()) # access the axis 
plt.show()



#%% Bonus video THE EYE of SAURON 

from matplotlib.patches import Ellipse  

# the sclera 
el1=Ellipse((0,0),width=45,height=28)
el1.set_facecolor((195/255,135/255,35/255))
# the cornea 


el2=Ellipse((0,0),width=25,height=25)
el2.set_facecolor((155/255,35/255,5/255))
# the pupil


el3=Ellipse((0,0),width=4,height=23)
el3.set_facecolor((55/255,15/255,10/255))


fig,ax=plt.subplots(subplot_kw={'aspect':'equal'},figsize=(12,12))
ax.add_artist(el1)
ax.add_artist(el2)
ax.add_artist(el3)

ax.set_xlim(-30,30)
ax.set_ylim(-15,15)
ax.set_axis_off()
plt.show()



# # Ellipse(xy, width, height, angle=0, **kwargs)[source]
# ellipse_outer = Ellipse((0, 0), 2, 1, angle=0, alpha=1, color = "orange")
# ellipse_middle = Ellipse((0, 0), 1, 1, angle=0, alpha=1, color = "red")
# ellipse_inner_edge = Ellipse((0, 0), 0.3, 0.8, angle=0, alpha=1, color = "yellow")
# ellipse_inner= Ellipse((0, 0), 0.2, 0.7, angle=0, alpha=1, color = "black")
 
# fig, ax = plt.subplots(figsize=(10,10))
 
# ax.add_artist(ellipse_outer)
# ax.add_artist(ellipse_middle)
# ax.add_artist(ellipse_inner_edge)
# ax.add_artist(ellipse_inner)
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.set_axis_off()










































































