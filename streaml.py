import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import streamlit as st

# extracting data from yahoo finance
st.title("Stock Prediction on Realtime Data")
start="2012-01-01"
end=st.text_input("Enter last closing day")
compname=st.text_input("Enter Stock Ticker",'AAPL')
df=data.DataReader(compname,"yahoo",start,end)
   

# data describing
st.subheader("DATASET") 
st.write(df) 

# data visualisation

st.subheader("Closing price trend")
fig=plt.figure(figsize=(16,8))
plt.plot(df.Close)
st.pyplot(fig) 

# Data Preprocessing

df["Date"]=df.index
# stock_data['row_num'] = np.arange(len(stock_data))
std_2=df.reset_index(drop=True)
std_2=std_2.drop(["Open","High","Low","Adj Close","Volume"],axis=1)
st.subheader("DATASET with index") 
st.write(std_2)

# normalizing the close/last price column
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df["Close"]).reshape(-1,1))
# splitting train and test data
training_size=int(len(df1)*0.7)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----15 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)
time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# modelfitting
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(15,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=50,batch_size=15,verbose=1)
# extracting the last index to get the test input
k=std_2[std_2["Date"]==str(end)].index.values
x=k[0]
new_set=std_2.iloc[x-15:x,:]
l=np.array(new_set[["Close"]])
x_input=scaler.fit_transform(l).reshape(1,-1) 
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=15
i=0
while(i<7):
    
    if(len(temp_input)>15):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1

st.write(scaler.inverse_transform(lst_output))
tildate=len(df1)
st.subheader("Prediction results")
fig=plt.figure(figsize=(16,8))
## Plotting 
# shift train predictions for plotting
day_new=np.arange(1,16)
day_pred=np.arange(16,23)
plt.xlabel("No.of days")
plt.ylabel("Close price of next 7 days")
plt.plot(day_new,scaler.inverse_transform(df1[tildate-15:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
st.pyplot(fig)

st.subheader("Future trend")
fig=plt.figure(figsize=(16,8))
df3=df1.tolist()
df3.extend(lst_output)
plt.xlabel("days")
plt.ylabel("Close price including  next 7 days")
plt.plot(scaler.inverse_transform(df3[tildate-20:]))
st.pyplot(fig)