import tensorflow as tf
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report # Importar para avaliação

diabetes = pd.read_csv('D:\GITHUB\machine-learning-studies\Redes Neurais para Classificação\calassification-for-diabetes\diabetes.csv')
print(diabetes.iloc[:, 8].value_counts(normalize=True))
X = diabetes.iloc[:, 0:8].values
y = diabetes.iloc[:, 8].values

# 1. Dividir PRIMEIRO
# Adicionar random_state para reprodutibilidade
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# 2. Criar e AJUSTAR (fit) o scaler APENAS nos dados de TREINO
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)

# 3. APENAS TRANSFORMAR os dados de TESTE com o scaler ajustado
X_test_scaled = sc.transform(X_test)

# 4. Salvar o scaler que foi AJUSTADO nos dados de treino
scaler_filename = 'D:\GITHUB\machine-learning-studies\Redes Neurais para Classificação\calassification-for-diabetes\scaler_diabetes.joblib'
joblib.dump(sc, scaler_filename)
print(f"Scaler (ajustado em X_train) salvo em {scaler_filename}")

# --- Construção e Compilação do Modelo (como antes) ---
classifier = tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Dense(units=400, activation='relu', input_shape=(8, )))
classifier.add(tf.keras.layers.Dropout(0.2))
classifier.add(tf.keras.layers.Dense(units=400, activation='relu'))
classifier.add(tf.keras.layers.Dropout(0.2))
classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='Adam', loss='binary_crossentropy', metrics = ['accuracy'])

# 5. Treinar o modelo com os dados de TREINO ESCALADOS
# Adicionar validation_data para monitorar performance no teste durante o treino
history = classifier.fit(X_train_scaled, y_train, epochs = 200,
                         validation_data=(X_test_scaled, y_test),
                         verbose=1) # verbose=1 para ver o progresso

# 6. Avaliar no conjunto de TESTE ESCALADO
print("\nAvaliação no conjunto de teste:")
loss, accuracy = classifier.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Obter predições no teste para relatório detalhado
y_pred_proba = classifier.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5) # Usando threshold 0.5 por enquanto

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['Não Diabetes (0)', 'Diabetes (1)']))

# 7. Salvar o modelo treinado
model_filename = 'D:\GITHUB\machine-learning-studies\Redes Neurais para Classificação\calassification-for-diabetes\model_diabetes.h5'
classifier.save(model_filename)
print(f"\nModelo treinado e salvo com sucesso em {model_filename}")