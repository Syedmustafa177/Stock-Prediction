import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.models import load_model
import streamlit as st  
from sentiment import Newsanalysis

news = Newsanalysis()


st.set_page_config(
    page_title= "Stock prediction app",
    page_icon="market-analysis.png"
)
# Define the stock symbol, start date, and end date
st.title("💹 Stock Trend prediction")

start = '2010-01-01'
end = st.text_input("📅 Enter Current Date",'2023-07-29')



user_input = st.text_input("Enter Stock Ticker", "WIT")

############################# news data #########################

news.newsa(user_input)

st.subheader("🤔 Sentiment Analysys")
result = news.newsa(user_input)
st.write(result)


# Retrieve the stock data
df = yf.download(user_input , start=start, end=end)

###########Describing the data ##########################################
st.subheader(f"📅 Data from {start} - {end}")
st.write(df.describe())
################################# VIsulaing the data #######################################
st.subheader("💸 Closing  Price VS Time Chart")
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)



st.subheader("💸 Closing  Price VS Time Chart with MA 100")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,"r")
plt.plot(df.Close)
st.pyplot(fig)


st.subheader("💸 Closing  Price VS Time Chart with MA100 & MA200")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,"r")
plt.plot(ma200,"g")
plt.plot(df.Close,"b")
st.pyplot(fig)


data_training = pd.DataFrame(df["Close"][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df["Close"][int(len(df)*0.70): int(len(df))])


scaler = MinMaxScaler(feature_range=(0,1))

data_traning_array = scaler.fit_transform(data_training)




################ my model ##################
model = load_model('keras.model.h5')

### testing part
past_100_days = data_training.tail(100)

final_df = past_100_days.append(data_testing,ignore_index = True)

input_data = scaler.fit_transform(final_df)

x_test =[]
y_test =[]

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test,y_test = np.array(x_test), np.array(y_test)


############ predictions ###############


y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader("🤓 predictions VS Original")
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,"b", label= "Original Price")
plt.plot(y_predicted,"r", label= "Predicted Price")
plt.xlabel('Time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig2)





























