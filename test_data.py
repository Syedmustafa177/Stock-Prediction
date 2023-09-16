
# df= df.reset_index()

# df = df.drop(["Date","Adj Close"],axis = 1)



# ma100 = df.Close.rolling(100).mean()
# ma200 = df.Close.rolling(200).mean()

# # plt.figure(figsize=(12,6))
# # plt.plot(df.Close)
# # plt.plot(ma100,"r")
# # plt.plot(ma200,"g")
# # plt.show()
# data_training = pd.DataFrame(df["Close"][0:int(len(df)*0.70)])
# data_testing = pd.DataFrame(df["Close"][int(len(df)*0.70): int(len(df))])

# # print(data_training.head)
# # print(data_testing.head)

# scaler = MinMaxScaler(feature_range=(0,1))

# data_traning_array = scaler.fit_transform(data_training)

# x_train =[]
# y_train =[]

# for i in range(100, data_traning_array.shape[0]):
#     x_train.append(data_traning_array[i-100 : i])
#     y_train.append(data_traning_array[i,0])


# x_train,y_train = np.array(x_train), np.array(y_train)


# ################################ML model################################################

# model = Sequential()
# model.add(LSTM(units=50,activation= "relu",return_sequences= True,input_shape=(x_train.shape[1], 1)))
# model.add(Dropout(0.2))
      

# model.add(LSTM(units=60,activation= "relu",return_sequences= True))
# model.add(Dropout(0.3))


# model.add(LSTM(units=80,activation= "relu",return_sequences= True))
# model.add(Dropout(0.4))

# model.add(LSTM(units=120,activation= "relu"))
# model.add(Dropout(0.5))

# model.add(Dense(units=1))



# # model.compile(optimizer='adam',loss='mean_squared_error')

# # model.fit(x_train,y_train, epochs= 50)

# # model.save('keras.model.h5')

# model = load_model('keras.model.h5')

# ###################################################################




# past_100_days = data_training.tail(100)

# final_df = past_100_days.append(data_testing,ignore_index = True)

# input_data = scaler.fit_transform(final_df)

# x_test =[]
# y_test =[]

# for i in range(100, input_data.shape[0]):
#     x_test.append(input_data[i-100: i])
#     y_test.append(input_data[i, 0])

# x_test,y_test = np.array(x_test), np.array(y_test)


# # print(x_test.shape)
# # print(y_test.shape)


# ##################### making predictions ##############################

# y_predicted = model.predict(x_test)


# scale_factor = 1/0.00641869
# y_predicted = y_predicted * scale_factor
# y_test = y_test * scale_factor


# plt.figure(figsize=(12,6))
# plt.plot(y_test,"b", label= "Original Price")
# plt.plot(y_predicted,"r", label= "Predicted Price")
# plt.xlabel('Time')
# plt.ylabel('price')
# plt.legend()
# plt.show()