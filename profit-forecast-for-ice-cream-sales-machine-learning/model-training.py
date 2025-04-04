import tensorflow as tf
import pandas as pd

sales_data = pd.read_csv('Your "SalesData.csv" directory here')

x_train = sales_data[['Temperature']].values
y_train = sales_data[['Revenue']].values

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units = 10, input_shape = [1]),
    tf.keras.layers.Dense(units = 1)
])

model.compile(optimizer = tf.keras.optimizers.Adam(0.001), loss = 'mean_squared_error')

model.fit(x_train, y_train, epochs = 300, verbose = 1)

model.save('profit_forecast_sales_model.h5')

print("Succesfuly saved model")