# src/train_mlp.py

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Carregar dados processados
df = pd.read_csv("../data/processed_flights.csv")

# Separar features e target
X = df.drop(["ARR_DELAY", "DELAYED"], axis=1)  # usar todas as colunas exceto targets
y = df["ARR_DELAY"]  # target numérica (regressão)

# Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Funções de ativação a testar
activations = ["relu", "tanh"]

for act in activations:
    print(f"\nTreinando MLP com função de ativação: {act}")

    # Criar modelo
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation=act))
    model.add(Dense(32, activation=act))
    model.add(Dense(16, activation=act))
    model.add(Dense(1, activation="linear"))  # regressão

    # Definir taxa de aprendizado manualmente
    optimizer = Adam(learning_rate=0.001)

    # Compilar modelo
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    # Treinar modelo
    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.2,
        verbose=1,
    )

    # Plotar gráficos de loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Treino")
    plt.plot(history.history["val_loss"], label="Validação")
    plt.title(f"Loss - Ativação {act}")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()
