import tensorflow as tf
import pandas as pd

sales_data = pd.read_csv('D:\Projetos\Python\Aprendendo Machine Learning\profit-forecast-for-ice-cream-sales-machine-learning\SalesData.csv')

x_train = sales_data[['Temperature']].values
y_train = sales_data[['Revenue']].values

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units = 10, input_shape = [1]),
    tf.keras.layers.Dense(units = 1)
])

model.compile(optimizer = tf.keras.optimizers.Adam(0.001), loss = 'mean_squared_error')

model.fit(x_train, y_train, epochs = 300, verbose = 1)

model.save('profit-forecast-for-ice-cream-sales-machine-learning/profit_forecast_sales_model.h5')

print("Succesfuly saved model")