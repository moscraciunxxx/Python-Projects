# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 06:23:03 2022

@author: moscr
"""

num=45.85
sng="cellar doors"

print("I would like to have "+ str(num)+ " "+sng )

print("I would like to have %g %s" %(num,sng))

print("I would like to have %9.3f %s " %(num,sng))

# f-string 

print(f"I would like to have {num} {sng}.")

print("I would like to have {n} {s}".format(n=num,s=sng))

#%% 
import string 
letters=string.ascii_lowercase 
letters
ordind='th1'

for i in range(len(letters)):
    if i==0 or i==20:
        ordind='st'
    elif i==1 or i==21:
        ordind='nd'
    elif i==2 or i==22:
        ordind='rd'
    else:
        ordind='th'
   # print("%s  is the %g%s letter of the alphabet"%(letters[i],i+1,ordind))
    print(f'{letters[i]} is the {i+1}{ordind} letter of the alphabet.')
    
    #%%
    
import matplotlib.pyplot as plt

plt.plot(3,4,'ro',label='red circle')
plt.plot(2,4,'bs',label='blue square')
plt.legend()
plt.show() # closes the current plot 
# plot a line 

plt.plot([0,1],[0,2],'r')
plt.show()

plt.plot([0,0],[1,2],'r')
plt.plot([0,1],[0,2],'g')
plt.legend(['red line','green line'])
plt.show()


import numpy as np

x=np.linspace(0, 3*np.pi,123)
y=np.sin(x)
plt.plot(x,y,'k.')
# plt.axis('square') 
plt.show()

#%% Subplot geometry

import matplotlib.pyplot as plt
import numpy as np
plt.subplot(1,2,1)
plt.plot(np.random.randn(10))

plt.subplot(1,2,2)
plt.plot(np.random.randn(10))
plt.show()

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.plot(np.random.randn(15))
    

fig,ax=plt.subplots(1,3,figsize=(10,3))
x=np.arange(25)
ax[0].plot(x,x**2,'b')

ax[1].plot(x,np.sqrt(x),'r')

ax[2].plot(x,x,'g')


fig, ax=plt.subplots(2,2)
ax[0,0].plot(np.random.randn(4,4))
ax[0,1].plot(np.random.randn(4,4))
ax[1,0].plot(np.random.randn(4,4))
ax[1,1].plot(np.random.randn(4,4))

plt.show()

#%% Create a 3x3 subplot  and populate the subplot using a for loop (use the method .flatten())

M=np.random.randint(1,11,size=(4,4))
print(M)
print(M.flatten())

fig,ax =plt.subplots(3,3,figsize=(8,8))
for i in ax.flatten():
    i.plot(np.random.randn(2,8))
    
plt.show() 


#%% Making the graphs look nicer
import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(-3,3,101)
plt.plot(x,x,label='y=x')
plt.plot(x,x**2,label='y=x**2')
plt.plot(x,x**3,label='y=x**3')


plt.xlabel('x')
plt.ylabel('y=f(x)')
plt.xlim([x[0],x[-1]])
plt.ylim([-10,10])
plt.title('A really awesome plot')

# plt.axis('axis')
#plt.gca().set_aspect('auto')
plt.gca().set_aspect(1/plt.gca().get_data_ratio())
plt.legend()
plt.show()

# or we could use plt.legend(['x','x**2','x**3'])

fig,ax=plt.subplots()

ax.plot(x,x,label='y=x')
ax.plot(x,x**2,label='y=x**2')
ax.plot(x,x**3,label='y=x**3',color=[.8,.1,.7])
ax.legend()

ax.set_xlabel('x')
ax.set_ylabel('y=f(x)')
ax.set_title('Another way to define a graph')
ax.set_xlim([-3,3])
ax.set_aspect(1/ax.get_data_ratio())
ax.grid()



plt.show()

#%% 
x=np.linspace(0,10,100)
for i in np.linspace(0,1,75):
    plt.plot(x,x**i,color=[i/2,0,i])
plt.show() 

#%% Adding annotations

import numpy as np
import matplotlib.pyplot as plt

x=np.arange(-5,6)
y=x**2

fig,ax=plt.subplots()
ax.plot(x,y,'ko-',markerfacecolor='w',markersize=15)

ax.annotate('Hello ',(x[3],y[3]),fontsize=15,arrowprops=dict(),
            xycoords='data',xytext=(0,10),horizontalalignment='center')

plt.show()

# Exercise

y=x**2+np.cos(x)*10
minpnt=np.argmin(y)
txt='min: (%g, %.2f)'%(x[minpnt],y[minpnt])

fig,ax=plt.subplots()
ax.plot(x,y,'ko-',markerfacecolor='purple',markersize=15)
ax.annotate(txt,(x[minpnt],y[minpnt]),fontsize=15,
            xycoords='data',xytext=(x[int(len(x)/2)],np.max(y)*.7),horizontalalignment='center',
            arrowprops=dict(color='purple',arrowstyle='wedge'))


ax.set_xlabel('x')
ax.set_ylabel('f(x)= x^2 + 10*cos(x)')


#%% Seaborn

import numpy as np
import seaborn as sns
import pandas as pd 
n=200
D=np.zeros((n,2))

D[:,0]=np.linspace(0,10,n)+np.random.randn(n)
D[:,1]=D[:,0]**2+np.random.randn(n)*10

sns.jointplot(x=D[:,0],y=D[:,1])
plt.show()

df=pd.DataFrame(data=D,columns=['var1','var2'])
sns.jointplot(x=df.columns[0],y=df.columns[1],data=df,kind='kde',color='purple')
plt.show()

#%%
x=np.linspace(-1,1,n)
y1=x**2
y2=np.sin(3*x)
y3=np.exp(-10*x**2)
plt.plot(y1,y2,'o')

sns.scatterplot(x=y1,y=y2,hue=y3,palette='rocket')
plt.show()


#%% regression plot of data of using seaborn 

sns.regplot(x=df.columns[0],y=df.columns[1],data=df,color='green')
plt.title(f'Regression of {df.columns[0]} on {df.columns[1]}')
plt.show()

#%% Images

import matplotlib.pyplot as plt
import numpy as np

m,n=3,5
M=np.random.randint(10,size=(m,n))
print(M)
plt.imshow(M)
for i in range(m):
    for j in range(n):
        plt.text(j,i,str(M[i,j]),horizontalalignment='center',fontsize=20,verticalalignment='center')

plt.colorbar() 
plt.show()


# image from the internet 

from imageio import imread
img=imread('https://upload.wikimedia.org/wikipedia/en/8/86/Einstein_tongue.jpg')
plt.imshow(img)
plt.title('That smart guy')

plt.hist(img.flatten(),bins=110)
plt.show()

plt.imshow(255-img,cmap="Greys",vmin=100,vmax=200)
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.show()

#%% Hilbert matrix 

# Create a Hilbert matrix (10x10) H[i,j]=1/(i+j-1)

n=10
H=np.zeros((n,n))
for i in range(1,n+1):
    for j in range(1,n+1):
        H[i-1,j-1]=1/(i+j-1)

# visualize the matric 

import seaborn as sns

sns.heatmap(H,vmin=0,vmax=.3)
plt.show()

#%% Export graphics in high and low resolution

import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(.5,5,15)
y1=np.log(x)
y2=2.5-np.sqrt(x)

plt.plot(x,y1,'bo-',label='log')
plt.plot(x,y2,'rs-',label='sqrt')
plt.legend()
plt.savefig('test.jpg')
plt.savefig('test.pdf')
plt.savefig('test.svg')
plt.show()
# get a better resolution figure in IPython console 

from IPython import display
display.set_matplotlib_formats('svg')
# matplotlib_inline.backend_inline.set_matplotlib_formats('svg')



































































































    