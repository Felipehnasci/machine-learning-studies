import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

model = tf.keras.models.load_model(r"D:\Projetos\Python\Aprendendo_Machine_Learning\forecast-for-rental-bikes-usage-machine-learning\model_bikes.h5")

st.set_page_config(page_title="Profit for sales 🍦", layout="centered")
st.markdown("<h1 style='text-align: center; color: #FF5733;'>Profit forecast for ice cream sales 🍦</h1>", unsafe_allow_html=True)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1804/1804293.png", width=100)

temp_c = st.sidebar.slider("Select the temperature (°C)", min_value=0, max_value=45, value=25)
hum = st.sidebar.slider("Select the humidity (%)", min_value=0, max_value=100, value=50)
windspeed = st.sidebar.slider("Select the windspeed (km/h)", min_value=0, max_value=100, value=10)

# Botão para realizar previsão
if st.sidebar.button("Predict revenue 💰"):

    input_data = np.zeros((1, 35))  # Criando um array com 35 valores, preenchendo com zero
    input_data[0, :3] = [temp_c, hum, windspeed]  # Substituindo os primeiros valores
    revenue = model.predict(input_data)

    
    
    st.markdown(
    f"""
    <div style="
        background-color: #d4edda;
        color: #155724;
        padding: 20px;
        border-radius: 10px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    ">
        💰 Estimated revenue: ${float(revenue):,.2f}
    </div>
    """, 
    unsafe_allow_html=True
)