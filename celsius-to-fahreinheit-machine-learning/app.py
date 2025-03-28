import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Carrega o modelo gerado pelo model-training.py
model = tf.keras.models.load_model('model_celsius_to_fahrenheit.h5')

# Titulo e input gerados pelo streamlit para melhor interação.
st.title('Conversor de Celsius para Fahrenheit.')
temp_c = st.number_input('Digite o numero em celsius para ser convertido em fahrenheit: ')

if temp_c is not None:
    temp_f = model.predict(np.array([temp_c]))
    st.write(f'O numero em graus Celsius convertido em Fahrenheit é: {float(temp_f):.2f}')


