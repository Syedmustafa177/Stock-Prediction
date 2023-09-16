import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sentiment import Newsanalysis

# Set the start and end dates for the stock data
start = '2010-01-01'
end = '2023-07-10'

# Get the stock ticker from user input
user_input = input("Enter Stock Ticker: ")

######################

news = Newsanalysis()
news.newsa(user_input)

result = news.newsa(user_input)

# Download historical stock data
df = yf.download(user_input, start=start, end=end)

# Preprocess the stock data
df = df[['Open', 'Close', 'High', 'Low']]  # Select relevant columns
df = df.fillna(method='ffill')  # Forward fill missing values

# Normalize the stock data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Define the sequence length for input data
sequence_length = 30

# Prepare the input data
data = []
target = []
for i in range(len(df_scaled) - sequence_length):
    data.append(df_scaled[i:i + sequence_length])
    target.append(df_scaled[i + sequence_length])

data = np.array(data)
target = np.array(target)

# Split the data into training and testing sets
split = int(0.8 * len(data))
x_train = data[:split]
y_train = target[:split]
x_test = data[split:]
y_test = target[split:]

# Retrieve the sentiment data
sentiment_data = result['sentiment'].values.reshape(-1, 1)  # Extract the sentiment column from result DataFrame

# Perform one-hot encoding on the sentiment data
encoder = OneHotEncoder()
sentiment_data_encoded = encoder.fit_transform(sentiment_data).toarray()

# Normalize the sentiment data
sentiment_data_scaled = scaler.transform(sentiment_data_encoded)

# Combine the stock data and sentiment data
combined_data_train = np.concatenate((x_train, sentiment_data_scaled[:split]), axis=1)
combined_data_test = np.concatenate((x_test, sentiment_data_scaled[split:]), axis=1)

# Define and train the neural network model
model = Sequential()
model.add(LSTM(64, input_shape=(combined_data_train.shape[1], combined_data_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(combined_data_train, y_train, epochs=10, batch_size=16)

# Evaluate the model on the testing set
loss = model.evaluate(combined_data_test, y_test)
print("Testing Loss:", loss)
