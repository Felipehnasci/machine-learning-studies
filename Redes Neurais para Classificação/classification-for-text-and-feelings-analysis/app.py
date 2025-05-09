import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import os # Para construir caminhos de forma segura

# --- Configuração de Caminhos ---
# Use caminhos relativos ou absolutos seguros
BASE_DIR = r'D:\GITHUB\machine-learning-studies\Redes Neurais para Classificação\classification-for-text-and-feelings-analysis'
MODEL_PATH = os.path.join(BASE_DIR, 'alexa_classifier_model.h5')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'alexa_vectorizer.joblib')
FEATURE_INFO_PATH = os.path.join(BASE_DIR, 'feature_info.joblib')

# --- Carregar Modelo, Vectorizer e Informações de Features ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    feature_info = joblib.load(FEATURE_INFO_PATH)
    num_dummy_features = feature_info['num_dummy_features']
    print("Modelo, Vectorizer e Infos carregados com sucesso.")
    # Opcional: Verificar input shape esperado pelo modelo
    expected_input_shape = model.input_shape[1]
    print(f"Modelo espera input shape: (None, {expected_input_shape})")
except Exception as e:
    st.error(f"Erro ao carregar arquivos necessários: {e}")
    print(f"Erro ao carregar arquivos: {e}") # Log no console também
    st.stop() # Impede a execução do restante do app se o carregamento falhar

# --- Interface Streamlit ---
st.set_page_config(page_title="Análise de Sentimento Alexa", layout="centered")
st.markdown("<h1 style='text-align: center; color: #1E90FF;'>Análise de Sentimento de Reviews (Alexa) 💬</h1>", unsafe_allow_html=True)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/732/732220.png", width=100) # Ícone Amazon/Alexa

st.markdown("---")
st.subheader("Digite a sua avaliação sobre a Alexa:")
user_input_text = st.text_area("Avaliação:", height=150, placeholder="Ex: 'Adorei o produto, funciona perfeitamente!' ou 'Muito difícil de configurar.'")

if st.button("Analisar Sentimento ✨"):
    if user_input_text and user_input_text.strip(): # Verifica se não está vazio ou só com espaços
        try:
            # 1. Processar o texto de entrada usando o Vectorizer carregado
            text_features = vectorizer.transform([user_input_text]).toarray()
            num_text_features_input = text_features.shape[1]
            print(f"Texto vetorizado com shape: {text_features.shape}")

            # 2. Criar as features dummy (um array de zeros)
            # O modelo foi treinado com [dummies, text_features]
            dummy_zeros = np.zeros((1, num_dummy_features))
            print(f"Features dummy (zeros) com shape: {dummy_zeros.shape}")

            # 3. Combinar as features na ordem correta [dummies, text]
            model_input = np.concatenate((dummy_zeros, text_features), axis=1)
            print(f"Input final para o modelo com shape: {model_input.shape}")

            # Verificar se o shape final bate com o esperado pelo modelo
            if model_input.shape[1] != expected_input_shape:
                 st.error(f"Erro de dimensionalidade! Shape do input: {model_input.shape[1]}, Esperado: {expected_input_shape}. Verifique o vectorizer e o número de dummies.")
                 st.stop()

            # 4. Fazer a predição
            prediction_proba = model.predict(model_input)
            probability_positive = prediction_proba[0][0] # Probabilidade de ser positivo (feedback=1)

            # 5. Definir o resultado com base no threshold (0.5)
            is_positive = (probability_positive > 0.5)

            # 6. Exibir o resultado
            st.markdown("---")
            st.subheader("Resultado da Análise:")
            if is_positive:
                st.success("Sentimento: POSITIVO 👍")
            else:
                st.error("Sentimento: NEGATIVO 👎")

            # Exibir a probabilidade formatada
            st.write(f"Confiança (Positivo): {probability_positive * 100:.2f}%")
            st.progress(float(probability_positive)) # Barra de progresso visual

        except Exception as e:
            st.error(f"Erro durante o processamento ou predição: {e}")
            print(f"Erro na predição: {e}") # Log no console
    else:
        st.warning("Por favor, digite uma avaliação para analisar.")

st.markdown("---")
st.info("Este modelo foi treinado para classificar avaliações sobre produtos Alexa como positivas (1) ou negativas (0).")