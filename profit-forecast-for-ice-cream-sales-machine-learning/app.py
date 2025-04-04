import tensorflow as tf
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Carregar o modelo salvo
model = tf.keras.models.load_model('profit_forecast_sales_model.h5')

# Configurar página do Streamlit
st.set_page_config(page_title="Profit for sales 🍦", layout="centered")

# Título
st.markdown("<h1 style='text-align: center; color: #FF5733;'>Profit forecast for ice cream sales 🍦</h1>", unsafe_allow_html=True)

# Sidebar com opções
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1804/1804293.png", width=100)
temp_c = st.sidebar.slider("Select the temperature (°C)", min_value=0, max_value=45, value=25)

# Botão para realizar previsão
if st.sidebar.button("Predict revenue 💰"):

    temp_f = model.predict(np.array([temp_c]))
    
    
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
        💰 Estimated revenue: ${float(temp_f):,.2f}
    </div>
    """, 
    unsafe_allow_html=True
)

