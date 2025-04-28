import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#CHANGE THE PATH TO YOUR FILES
house = pd.read_csv("YOUR_PATH_TO_FILE/house_data.csv")
house['date'] = pd.to_datetime(house['date'])
house['date_numeric'] = (house['date'] - house['date'].min()).dt.days

selected_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement']
X = house[selected_features]
Y = house['price']

scaler = MinMaxScaler()
X_scaler = scaler.fit_transform(X)

Y = Y.values.reshape(-1, 1)
Y_scaler = scaler.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaler, Y_scaler, test_size=0.25)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100, activation='relu', input_shape=(7, )))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))

model.compile(optimizer='Adam', loss='mean_squared_error')

model.fit(X_train, Y_train, epochs = 5, batch_size = 50, validation_split = 0.2)

model.save('profit-for-house-prices\model_houses.h5')

print("Modelo treinado e salvo com sucesso!")