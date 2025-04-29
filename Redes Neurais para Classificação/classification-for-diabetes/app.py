import tensorflow as tf
import numpy as np
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler

#CHANGE THE PATH TO YOUR FILES
try:
    model_path = r"D:\GITHUB\machine-learning-studies\Redes Neurais para Classificação\classification-for-diabetes\model_diabetes.h5"
    scaler_path = r"D:\GITHUB\machine-learning-studies\Redes Neurais para Classificação\classification-for-diabetes\scaler_diabetes.joblib"
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    print("Scaler Mean:", scaler.mean_)
    print("Scaler Scale (Std Dev):", scaler.scale_)# Carrega o scaler
    print("Modelo e Scaler carregados com sucesso.")
except Exception as e:
    st.error(f"Erro ao carregar o modelo ou scaler: {e}")
    st.stop() # Para a execução se não conseguir carregar
    
    
# Configuração da página no Streamlit
st.set_page_config(page_title="Previsão de Diabetes 🩺", layout="centered")
st.markdown("<h1 style='text-align: center; color: #FF5733;'>Previsão de Diabetes</h1>", unsafe_allow_html=True)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1804/1804293.png", width=100)

# Criar sliders com keys únicas
pregnancies = st.number_input("Número de Gravidezes", min_value=0, value=1, step=1, key="slider_pregnancies")
glucose = st.number_input("Nível de Glicose", min_value=0.0, value=10.0, step=1.0, key="slider_glucose")
bloodpressure = st.number_input("Pressão Sanguínea (mmHg)", min_value=0.0, value=70.0, step=1.0, key="slider_bloodpressure")
skinthickness = st.number_input("Espessura da Pele (mm)", min_value=0.0, value=20.0, step=0.1, key="slider_skinthickness")
insulin = st.number_input("Nível de Insulina (mu U/ml)", min_value=0.0, value=80.0, step=1.0, key="slider_insulin")
BMI = st.number_input("Índice de Massa Corporal (IMC)", min_value=0.0, value=25.0, step=0.1, key="slider_BMI")
diabetesPedigreeFunction = st.number_input("Função Diabetes Pedigree", min_value=0.0, value=0.5, step=0.001, format="%.3f", key="slider_diabetesPedigreeFunction")
age = st.number_input("Idade", min_value=0, value=30, step=1, key="slider_age")


if st.button("Prever Diabetes 🩺"):
    
    # Criar array com os dados de entrada
    input_data = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, BMI, diabetesPedigreeFunction, age]])

    # Escalador para os dados de entrada
    try:
        input_data_scaled = scaler.transform(input_data) # Use .transform()
    except Exception as e:
         st.error(f"Erro ao escalar os dados de entrada: {e}")
         st.stop()

    try:
        prediction_proba = model.predict(input_data_scaled) # Retorna a probabilidade
        prediction = (prediction_proba > 0.6)[0][0] # Threshold para obter True/False
                                                # [0][0] para pegar o valor booleano do array
    except Exception as e:
        st.error(f"Erro ao realizar a predição: {e}")
        st.stop()
    # Fazer previsão
    st.markdown("---")
    st.subheader("Resultado da Predição:")

    if prediction:
        st.error("Resultado: POSITIVO para Diabetes")
        st.warning("Recomenda-se consultar um médico.")
    else:
        st.success("Resultado: NEGATIVO para Diabetes")
        st.info("Continue mantendo hábitos saudáveis.")

    probabilidade_decimal = prediction_proba[0][0]
    st.write(f"Probabilidade de ter diabetes: {probabilidade_decimal * 100:.2f}%")