# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 06:40:07 2022

@author: moscr
"""
import glob
import re 
import matplotlib.pyplot as plt 
pyfiles=glob.glob('*ipy')

# import each file 

allfiles=[]

for fil in pyfiles:
    fid=open(fil,'r')
    content=fid.read()
    allfiles.append(content)
    

#%% another method

allfiles=[]
for fil in pyfiles:
    with open(fil,'r') as fid:
        allfiles.append(fid.read())
        
        

#%%  Identify function calls
# find workds that end with e
s='My name is Vitalie and I like to eat chocolate'
re.findall(r'\w*e\b',s)

# find words that start with V 
re.findall(r'\bV\w*',s)


t='"alist=list( range(5,11) )\\n"'
re.findall(r'\w*\( ?\b',t)
 

funwords=re.findall(r'\w*\( ?\b',allfiles[0])
funwords=[i.strip() for i in funwords]  # remove all the spaces at the end of the words
funwords=[i.strip()[:-1] for i in funwords] # remove the paranthesis too 
funwords=[i for i in funwords if len(i)>0]  # remove the empty items 
#%% 



allfuncalls=[]
fi=0

for thisfile in allfiles:
        
    funwords=re.findall(r'\w*\( ?\b',thisfile)
    funwords=[i.strip()[:-1] for i in funwords] # remove the paranthesis too 
    funwords=[i for i in funwords if len(i)>0]  # remove the empty items
        
    
    for f in funwords:
        allfuncalls.append([f,pyfiles[fi][:-4]])
        
    # increment the file counter 
    fi+=1

#%% the same thing but now let's use enumerate 


allfuncalls=[]

for fi,thisfile in enumerate(allfiles):  # enumerate function returns a tuple 
        
    funwords=re.findall(r'\w*\( ?\b',thisfile)
    funwords=[i.strip()[:-1] for i in funwords] # remove the paranthesis too 
    funwords=[i for i in funwords if len(i)>0]  # remove the empty items
        
    
    for f in funwords:
        allfuncalls.append([f,pyfiles[fi][:-4]])
  


#%% Create an alphabetized function index 

uniquenames=[]
for i in allfuncalls:
    if i[0] not in uniquenames:
        uniquenames.append(i[0])
        
        



indexdict={}
for i in allfuncalls:
    # create a new key if it doesn't exist 
    if i[0] not in indexdict:
        indexdict[i[0]]=[]
        
    # add the filename to the value list 
    if i[1] not in indexdict[i[0]]:
        indexdict[i[0]].append(i[1])
        

#%% 
fid=open('function_index.txt','w')
for key in sorted(indexdict):  # alphabetize the keys 
    
    # function call in the first column 
    fid.write(f'{key};')
    # files in the second cell 
    temp_str=str(indexdict[key]).replace('[','').replace(']','').replace("'",'')
    
    fid.write(f'{temp_str}')
    fid.write('\n')
fid.close()
        


#%% 



indexdict={}
for tmp_funcall,tmp_filename in allfuncalls:
    
    # force lowercase filename: 
    tmp_filename=tmp_filename.lower()
    # create a new key if filename doesn't exist 
    if tmp_funcall not in indexdict:
        indexdict[tmp_filename]=[]
        
    # add the filename to the value list 
    if tmp_funcall not in indexdict[tmp_filename]:
        indexdict[tmp_filename].append(tmp_funcall)
        
        #%%
        

fid=open('file_index.txt','w')
for key in sorted(indexdict):  # alphabetize the keys 
    
    # files in the first column 
    fid.write(f'{key};')
    # function calls  in the second cell 
    temp_str=str(sorted(indexdict[key])).replace('[','').replace(']','').replace("'",'')
    
    fid.write(f'{temp_str}')
    fid.write('\n')
fid.close()
        


#%% Bonus video: which file has the most points ?
total=[0]*len(allfiles)

# loop through all the files 
for fidx,fil in enumerate(allfiles):
    

# loop through all characters and get their score 
    for c in fil:
       

# sum all of the character scores 
        total[fidx]+=ord(c)
# normalize to character length
    total[fidx]/=len(fil)

for i in range(len(allfiles)):
    print(f'{pyfiles[i][:-4]} has {total[i]:.4f} points')
    
plt.bar(range(len(total)),total)

xlabels=[i[:-4] for i in pyfiles]

plt.ylabel('Unicode score',fontsize=15)
plt.xlabel('Filenames ',fontsize=15)
plt.xticks(range(len(total)),labels=xlabels,rotation=60)
plt.ylim([min(total)-3,max(total)+3])
plt.show()






















































