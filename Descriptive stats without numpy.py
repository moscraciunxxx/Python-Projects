# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 01:08:47 2022

@author: moscr
"""

import numpy as np
from scipy import stats



#%% mean and median 
N=25
data=list(np.random.randn(N))

mymean=sum(data)/len(data)

# print(data)
# data.sort()
# print(data)

if N%2==1:
    mymedian=data[N//2]  # floor division  
else:
    mymedian=(data[N//2+1]*data[N//2])/2  
    
print(mymedian)


#%% Frequencies table 

l=[3,5,'q_o',-55,np.pi]

for i,j in enumerate(l):
    print(i,j)


data=list(np.random.randint(1,15,60))


# unique values 
uniquevalues=set(data)
valcounts=[0]*len(uniquevalues)
# the counts at which values appear in the data 

for d in data:
    for i,u in enumerate(uniquevalues):
       if d==u:
           valcounts[i]+=1
           
print(uniquevalues)
print(valcounts)          
        

#%% list comprehension 

for d in data:
    idx=[i for i,u in enumerate(uniquevalues) if d==u]
    valcounts[idx[0]]+=1

    
print(uniquevalues)
print(valcounts)

# convert the lists into a dictionary 
uniquevalues=list(set(data))
table={}
for i in range(len(uniquevalues)):
    table[uniquevalues[i]]=valcounts[i]

print(table)



#%% Mode 

maxcount=0
mode=[]
for i in table.items():
    if i[1]>=maxcount:
        maxcount=i[1]
        
        mode.append(i[0])


print('Mode %s appears %s times '%(mode,maxcount))



maxcount2=max(list(table.values()))

mode2=[k for k,v, in table.items() if v==maxcount2]

print(mode2)


 
stats.mode(data)


#%% Standard deviation 

N=101
data=list(np.random.poisson(2,N))


meanval=sum(data)/len(data)
summation=sum([(d-meanval)**2 for d in data])

std=(summation/(N-1))**.5

# std=(summation/(N))**.5 biased STD 
print(std)
print(np.std(data,ddof=1))  # denominator degrees of freedom 

#%% Bonus : Create a csv report file 

def mymean(data):
    return sum(data)/len(data)
    
    
def mymedian(data):
    
    data.sort()
    N=len(data)
        
    if N%2==1:
        med=data[N//2]  # floor division  
    else:
        med=(data[N//2+1]*data[N//2])/2  
        
    return med   
    
def mymode(data):
    # unique values 
    uniquevalues=list(set(data))
    valcounts=[0]*len(uniquevalues) # the counts at which values appear in the data   
    
    # list comprehension 

    for d in data:
        idx=[i for i,u in enumerate(uniquevalues) if d==u]
        valcounts[idx[0]]+=1
    table={}
    for i in range(len(uniquevalues)):
        table[uniquevalues[i]]=valcounts[i]  
        
    maxcount2=max(list(table.values()))    
    mode2=[k for k,v, in table.items() if v==maxcount2]
    return mode2 
        

def mystd(data):
        
    meanval=sum(data)/len(data)
    summation=sum([(d-meanval)**2 for d in data])    
    std=(summation/(N-1))**.5
    return std 
    

N=60

data=np.random.randint(10,80,N)
fid=open('stats.csv','w')
fid.write('mean = ' + str(mymean(data))+'\n')
fid.write('median = ' + str(mymedian(data))+'\n')
fid.write('mode = ' + str(mymode(data))+'\n')
fid.write('std = ' + str(mystd(data))+'\n')

fid.close()



























































