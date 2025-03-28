import tensorflow as tf
import pandas as pd
import numpy as np


# MUDE PARA O DIRETORIO DO SEU ARQUIVO "Celsius-to-Fahrenheit.csv"
# CHANGE THE DIRECTORY OF YOUR "Celsius-to-Fahrenheit.csv" ARCHIVE
temperature_df = pd.read_csv(r'D:Your directory here\Celsius-to-Fahrenheit.csv')

x_train = temperature_df['Celsius'].values.reshape(-1, 1)
y_train = temperature_df['Fahrenheit'].values.reshape(-1, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.5), loss='mean_squared_error')

model.fit(x_train, y_train, epochs=300, verbose=1)

model.save('model_celsius_to_fahrenheit.h5')

print("Modelo treinado e salvo com sucesso! // Model has been succesfuly trained !")
