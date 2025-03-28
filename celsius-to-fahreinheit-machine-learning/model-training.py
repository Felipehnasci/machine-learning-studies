import tensorflow as tf
import pandas as pd
import numpy as np

# Carregar os dados
temperature_df = pd.read_csv(r'D:\Projetos\Python\Aprendendo Machine Learning\celsius-to-fahreinheit-machine-learning\Celsius-to-Fahrenheit.csv')

# Definir os dados de treino
x_train = temperature_df['Celsius'].values.reshape(-1, 1)
y_train = temperature_df['Fahrenheit'].values.reshape(-1, 1)

# Criar o modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compilar o modelo
model.compile(optimizer=tf.keras.optimizers.Adam(0.5), loss='mean_squared_error')

# Treinar o modelo
model.fit(x_train, y_train, epochs=300, verbose=1)

# Salvar o modelo treinado
model.save('modelo_celsius_para_fahrenheit.h5')

print("Modelo treinado e salvo com sucesso!")