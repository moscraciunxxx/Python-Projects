# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:20:58 2022

@author: moscr
"""
from Historic_Crypto import HistoricalData
from Historic_Crypto import Cryptocurrencies

import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA




#%% Import and average data from one coin

# Cryptocurrencies().find_crypto_pairs()

eth=HistoricalData('ETH-EUR', 60*60*24, '2020-01-01-00-00','2021-12-01-00-00').retrieve_data()

eth.plot.line();  # pandas method 

eth[['low','high','open','close']].plot.line()

eth.plot.line(x='low',y='high',marker='o')

#%% 

sns.pairplot(eth)

eth[['low','high','open','close']].mean()  # the average of rows per column 

# but we want to average over the column within each row 


eth['average price']=eth[['low','high','open','close']].mean(axis=1)

eth['average price'].plot(logy=True)  # logarithmic scale 


#%% Create a dataframe of selected coins 

for item in Cryptocurrencies().find_crypto_pairs()['id'].tolist():
    if 'EUR' in item: print(item)
    
    
coins2eval=['BTC-EUR','ETH-EUR','SHIB-EUR','DOGE-EUR','XLM-EUR','ALGO-EUR','LINK-EUR']

coinpricesD={}
for ticker in coins2eval:
    # import the historical data
    temp=HistoricalData(ticker, 60*60*24, '2020-01-01-00-00','2021-12-01-00-00').retrieve_data()

    
    # take the average price 
    aver=temp[['low','high','open','close']].mean(axis=1)   
    
    # store the price in the dict 
    coinpricesD[ticker]=aver
    
#%% 
coinprices=pd.DataFrame(coinpricesD)

# scale and plot the data 

coinprices.plot()

coinpricesScaled=(coinprices-coinprices.min())/(coinprices.max()-coinprices.min())

coinpricesScaled.plot(title='Scaled to max ', fontsize=20,figsize=(15,8))

#%% Scaled to 1 at 11th  of May


coinpricesScaled=(coinprices-coinprices.min())/(coinprices.loc['2021-05-11'].values[:]-coinprices.min())

coinpricesScaled.plot(title='Scaled to 1 at 11 of May 2021 ', fontsize=20,figsize=(15,8))

#%% Data dimensionality via PCA 


# percent variance explained (explained variance ratio)
coinprices.corr()
# for the scaled data the corr is going to be the same because the formula already removes the 
# scale of the data 
coinpricesScaled.corr()



# PCA 

pca=PCA()
pca.fit(coinprices.dropna())


pcaS=PCA()
pcaS.fit(coinpricesScaled.dropna())

coinpricesScaledCentered=coinpricesScaled.sub(coinpricesScaled.mean(axis=1),axis=0)# .plot()
pcaM=PCA()
pcaM.fit(coinpricesScaledCentered.dropna())


plt.plot(100*pca.explained_variance_ratio_,'o-',label="Raw data")
# plt.plot(100*pcaS.explained_variance_ratio_,'o-',label="Scaled data")
plt.xticks(range(pca.n_components_))
plt.xlabel('Components')
plt.ylabel('Percent variance explained')
plt.legend()
plt.title("Scree plot")


plt.show()

# the plot show that the market is driven by a single factor (BTC here)
# the nth component is accounting for a certain percent of the variance 


#%% Simulating DCA investments 

# DCA - dollar cost averaging 
# Lump-sum 

# simulation 1

dailyInvest=10

# which coin to simulate    
whichCoin='ETH-EUR'

# initialize our investment amounts 

euroInvest=0
coinInvest=0


# Loop through days

for dayi in range(coinprices.shape[0]):
    # buy some coin 
    coin=dailyInvest/coinprices[whichCoin][dayi]
    
    # add the totals 
    euroInvest+=dailyInvest
    coinInvest+=coin
    

# compute the final value in euros of our investment 
eurosAtEnd=coinInvest*coinprices[whichCoin][-1]

print(f'Total euro invested :\u20ac {euroInvest:,.2f}')
print(f'Total {whichCoin[:-4]} purchased : {coinInvest:.7f}')
print(f'End result is :\u20ac {eurosAtEnd:,.2f}')


#%% Simulation 2 



dailyInvestUp=7
dailyInvestDn=15


# which coin to simulate    
whichCoin='ETH-EUR'

# initialize our investment amounts 

euroInvest=0
coinInvest=0


# Loop through days

for dayi in range(1,coinprices.shape[0]):
    # buy some coin 
    if (coinprices[whichCoin][dayi] > coinprices[whichCoin][dayi-1]):
        coin=dailyInvestUp/coinprices[whichCoin][dayi]
        euroInvest+=dailyInvestUp
    else: # price went down 
        coin=dailyInvestDn/coinprices[whichCoin][dayi]
        euroInvest+=dailyInvestDn
    
    # add the totals 
    
    coinInvest+=coin
    
    

# compute the final value in euros of our investment 
eurosAtEnd=coinInvest*coinprices[whichCoin][-1]

print(f'Total euro invested :\u20ac {euroInvest:,.2f}')
print(f'Total {whichCoin[:-4]} purchased : {coinInvest:.7f}')
print(f'End result is :\u20ac {eurosAtEnd:,.2f}')



#%% Simulation 3

dailyInvestUp=10


# which coin to simulate    
whichCoin='ETH-EUR'

# initialize our investment amounts 

euroInvest=0
coinInvest=0
percChange=[0]*coinprices.shape[0]


# Loop through days

for dayi in range(1,coinprices.shape[0]):
    # compute the percent change from the previous day 
    percChange[dayi]=100* (coinprices[whichCoin][dayi] - coinprices[whichCoin][dayi-1])/coinprices[whichCoin][dayi-1] 
    
    # # buy some coin 
    # if percChange >0: # price went up 
    #     coin=dailyInvest/coinprices[whichCoin][dayi]
    #     euroInvest+=dailyInvest
        
    # else: # price went down 
    #     toInvest=dailyInvest*-percChange
    #     coin=toInvest/coinprices[whichCoin][dayi]
    #     euroInvest+=toInvest
    if percChange[dayi] >0: # price went up 
        toInvest=dailyInvest
    else: # price went down 
         toInvest=dailyInvest*-percChange[dayi]
         
    # add the totals  
    coin=toInvest/coinprices[whichCoin][dayi]
    euroInvest+=toInvest
    coinInvest+=coin
    
# compute the final value in euros of our investment 
eurosAtEnd=coinInvest*coinprices[whichCoin][-1]

print(f'Total euro invested :\u20ac {euroInvest:,.2f}')
print(f'Total {whichCoin[:-4]} purchased : {coinInvest:.7f}')
print(f'End result is :\u20ac {eurosAtEnd:,.2f}')

#%% Bonus Video  Which coin you should have bought 

dailyInvest=10
# initialize the investments
euroInvest={}
coinInvest={}
# loop through the days 
for dayi in range(coinprices.shape[0]):
    
    # loop over all of the coins 
    for coinname in coins2eval:
        # initialize the investments amount 
        if dayi ==0:
            
            euroInvest[coinname]=0
            coinInvest[coinname]=0
    # how much coin did we buy on this day 
        coin =dailyInvest/coinprices[coinname][dayi]
    # add the totals 
        if np.isnan(coin): continue
    
        euroInvest[coinname] +=dailyInvest
        coinInvest[coinname] +=coin
    
for coinname in coins2eval:
    # how much money in euros 
    
    eurosAtEnd=coinInvest[coinname]*coinprices[coinname][-1]
    
    print(f'{coinname[:-4]:>14}: \u20ac{euroInvest[coinname]:,.2f} \u21e8 \u20ac{eurosAtEnd:>9,.2f}')






























































































































































