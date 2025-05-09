import pandas as pd
import numpy as np
import tensorflow as tf # Importar tensorflow
import joblib # Importar joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report # Para avaliar melhor

# --- Seu código existente para carregar e pré-processar os dados ---
df_alexa = pd.read_csv(r'D:\GITHUB\machine-learning-studies\Redes Neurais para Classificação\classification-for-text-and-feelings-analysis\amazon-alexa.tsv', sep = r'\t', engine='python') # Adicionado engine='python' para evitar warnings com \t

# Tratamento inicial
df_alexa = df_alexa.drop(['date', 'rating'], axis = 1)

# Guarda o número de colunas dummy para usar depois
variation_dummies = pd.get_dummies(df_alexa['variation'], drop_first=False) # drop_first=False garante que todas as variações estão presentes
num_dummy_features = variation_dummies.shape[1] # Guarda o número de colunas dummy
print(f"Número de features dummy de 'variation': {num_dummy_features}")

df_alexa.drop(['variation'], axis = 1, inplace=True)
df_alexa = pd.concat([df_alexa, variation_dummies], axis = 1)

# Preencher NaNs e criar/ajustar o Vectorizer
df_alexa['verified_reviews'] = df_alexa['verified_reviews'].fillna('') # Tratar NaNs ANTES de vetorizar
vectorizer = CountVectorizer()
alexa_countvectorizer = vectorizer.fit_transform(df_alexa['verified_reviews'])

# Salvar o Vectorizer ajustado
vectorizer_filename = r'D:\GITHUB\machine-learning-studies\Redes Neurais para Classificação\classification-for-text-and-feelings-analysis\alexa_vectorizer.joblib'
joblib.dump(vectorizer, vectorizer_filename)
print(f"Vectorizer salvo em {vectorizer_filename}")

# Continuar o pré-processamento para o treino
df_alexa.drop(['verified_reviews'], axis = 1, inplace=True)
reviews = pd.DataFrame(alexa_countvectorizer.toarray())
# É crucial garantir que as colunas do DataFrame 'reviews' sejam strings se for concatenar por nome
# Ou, mais simples, apenas concatenar os arrays numpy mais tarde ao preparar X
num_text_features = reviews.shape[1]
print(f"Número de features de texto (CountVectorizer): {num_text_features}")

# Preparar X e y - IMPORTANTE A ORDEM!
# As colunas em df_alexa neste ponto são ['feedback', dummy_col_1, ..., dummy_col_n]
# Precisamos combinar [dummy_col_1, ..., dummy_col_n] com [review_col_0, ..., review_col_m]
X_dummies = df_alexa.drop('feedback', axis=1).values 
X_reviews = reviews.values 

X = np.concatenate((X_dummies, X_reviews), axis=1) 
y = df_alexa['feedback'].values 

print(f"Shape final de X: {X.shape}") 


feature_info = {'num_dummy_features': num_dummy_features}
feature_info_filename = r'D:\GITHUB\machine-learning-studies\Redes Neurais para Classificação\classification-for-text-and-feelings-analysis\feature_info.joblib'
joblib.dump(feature_info, feature_info_filename)
print(f"Informações de features salvas em {feature_info_filename}")


# --- Seu código existente para dividir treino/teste ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

print(f"Input shape esperado pelo modelo: ({X_train.shape[1]},)") # Verifica o input shape

# --- Seu código existente para construir, compilar e treinar o modelo ---
classifier = tf.keras.models.Sequential()
# Use a dimensão correta obtida de X_train.shape[1]
classifier.add(tf.keras.layers.Dense(units = 400, activation='relu', input_shape=(X_train.shape[1],)))
classifier.add(tf.keras.layers.Dense(units = 400, activation='relu'))
classifier.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))

classifier.compile(optimizer='Adam', loss='binary_crossentropy', metrics = ['accuracy'])

# Não precisa converter X_train para int se ele já for numérico (CountVectorizer retorna int)
# X_train = X_train.astype(int) # Provavelmente desnecessário, Keras lida com float/int

epochs_hist = classifier.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test)) # Adiciona validation_data

# --- Avaliação ---
print("\nAvaliação no conjunto de teste:")
loss, accuracy = classifier.evaluate(X_test, y_test, verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")

y_pred_test_proba = classifier.predict(X_test)
y_pred_test = (y_pred_test_proba > 0.5)

print("\nRelatório de Classificação (Teste):")
# Feedback 1 é positivo, 0 é negativo
print(classification_report(y_test, y_pred_test, target_names=['Negativo (0)', 'Positivo (1)']))


# --- Salvar o Modelo Treinado ---
model_filename = r'D:\GITHUB\machine-learning-studies\Redes Neurais para Classificação\classification-for-text-and-feelings-analysis\alexa_classifier_model.h5'
classifier.save(model_filename)
print(f"\nModelo treinado e salvo com sucesso em {model_filename}")