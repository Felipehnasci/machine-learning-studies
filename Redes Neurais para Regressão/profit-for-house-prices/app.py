import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

#CHANGE THE PATH TO YOUR FILES
model = tf.keras.models.load_model(r"YOUR_PATH_TO_FILE\model_houses.h5")

# Configuração da página no Streamlit
st.set_page_config(page_title="Profit for houses prices 🍦", layout="centered")
st.markdown("<h1 style='text-align: center; color: #FF5733;'>Profit for house prices </h1>", unsafe_allow_html=True)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1804/1804293.png", width=100)

# Criar sliders com keys únicas
bedrooms = st.number_input("Select the quantity of badroons" , key="slider_bedrooms")
bathrooms = st.number_input("Select the quantity of bathroons",  key="slider_bathrooms")
sqft_living = st.number_input("Insert the area of living room in m²", key="slider_sqft_living")
sqft_lot = st.number_input("Insert the area of the terrain in m²", key="slider_sqft_lot")
floors = st.number_input("Select the quantity of floors" , key="slider_floors")
sqft_above = st.number_input("Insert the area above in m²" , key="slider_sqft_above")
sqft_basement = st.number_input("Insert the area of basement in m²", key="slider_sqft_basement")

# Suponha que temos um conjunto de preços de casas para ajuste do escalador
# Certifique-se de usar os preços reais do seu dataset
preco_treinamento = np.array([221900, 538000, 180000, 604000, 510000])  # Exemplo de preços reais
preco_treinamento = preco_treinamento.reshape(-1, 1)

# Ajustar o MinMaxScaler para os preços reais
scaler_price = MinMaxScaler()
scaler_price.fit(preco_treinamento)

# Botão de previsão
if st.sidebar.button("Predict price 💰"):
    
    # Criar array com os dados de entrada
    X_test_1 = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above, sqft_basement]])

    # Escalador para os dados de entrada
    scaler_input = MinMaxScaler()
    X_test_scaled_1 = scaler_input.fit_transform(X_test_1)

    # Fazer previsão
    price = model.predict(X_test_scaled_1)

    # Reverter a normalização do preço
    price_original = scaler_price.inverse_transform(price)

    # Exibir resultado formatado
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
        💰 Estimated price: ${float(price_original[0][0]):,.2f}
    </div>
    """, 
    unsafe_allow_html=True
    )
