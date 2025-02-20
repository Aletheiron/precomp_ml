import numpy as np
import pandas as pd

data=pd.read_csv('ICSA.csv')

#Extract change in claims
data.drop('DATE', axis=1, inplace=True)

data['shift']=data['ICSA'].shift(1)

#Moving to log return for comparability
data['change']=np.log((data['ICSA'])/data['shift'])
data.drop(index=0,inplace=True)
data.drop(['ICSA', 'shift'], axis=1, inplace=True)

#Prepare data for comparing

# for i in range(1,17):
    
#     data[f'day_before {i}']=data['change'].shift(i)

# data['target']=data['change'].shift(-1)
# data=data.dropna()

# #Save raw data
# data.to_csv('raw_data', header=False, index=False)

#Prepare data with precomputings

for i in range(1,17):
    
    data[f'day_before {i}']=data['change'].shift(i)


data=data.dropna()

#Simple moving average function

data['SMA_4']=data['change'].rolling(4).mean()
data['SMA_8']=data['change'].rolling(8).mean()
data['SMA_16']=data['change'].rolling(16).mean()

data['SMA_std_4']=data['change'].rolling(4).std()
data['SMA_std_8']=data['change'].rolling(8).std()
data['SMA_std_16']=data['change'].rolling(16).std()

data['target']=data['change'].shift(-1)

data=data.dropna()  
print(data.head())

#Save precompute data
#data.to_csv('pre_data', header=False, index=False)