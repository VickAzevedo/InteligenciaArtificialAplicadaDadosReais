# src/train_mlp.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

PROCESSED_PATH = os.path.join("data", "processed", "processed_flights.csv")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


LEARNING_RATES = [0.001, 0.005]
EPOCHS = 30
BATCH_SIZE = 32
ACTIVATIONS = ["relu", "tanh", "sigmoid"]

def load_data(path=PROCESSED_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo processado não encontrado: {path}")
    df = pd.read_csv(path)
    if "Delayed" not in df.columns:
        raise KeyError("Coluna 'Delayed' não encontrada no dataset processado.")
    return df

def build_mlp(input_dim, activation="relu", neurons=[128, 64, 32]):
    model = Sequential()
    # 3 camadas ocultas
    model.add(Dense(neurons[0], activation=activation, input_dim=input_dim))
    model.add(Dense(neurons[1], activation=activation))
    model.add(Dense(neurons[2], activation=activation))
    # saída binária
    model.add(Dense(1, activation="sigmoid"))
    return model

def run_experiment(df):
    X = df.drop(columns=["Delayed"])
    y = df["Delayed"].values

    print("Total registros:", len(df))
    print("Número de atributos (antes do scaling):", X.shape[1])
    print("Distribuição do target:\n", df["Delayed"].value_counts().to_dict())


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Tamanho treino/teste:", X_train.shape[0], X_test.shape[0])

    summary_rows = []
    for lr in LEARNING_RATES:
        for activation in ACTIVATIONS:
            print(f"\n--- Treinando: activation={activation} | lr={lr} ---")
            model = build_mlp(input_dim=X_train.shape[1], activation=activation)
            optimizer = Adam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=2
            )

            print("Vetor de loss (treino):")
            print(history.history["loss"])
            print("Vetor de loss (val):")
            print(history.history["val_loss"])

            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            y_prob = model.predict(X_test).ravel()
            y_pred = (y_prob >= 0.5).astype(int)

            cm = confusion_matrix(y_test, y_pred)
            cr = classification_report(y_test, y_pred, digits=4)
            acc = accuracy_score(y_test, y_pred)


            prefix = os.path.join(OUTPUT_DIR, f"mlp_act-{activation}_lr-{str(lr).replace('.', '')}")
            # loss
            plt.figure(figsize=(8,5))
            plt.plot(history.history["loss"], label="Treino")
            plt.plot(history.history["val_loss"], label="Validação")
            plt.title(f"Loss por época - act={activation} lr={lr}")
            plt.xlabel("Época"); plt.ylabel("Loss (binary_crossentropy)")
            plt.legend(); plt.grid(True)
            plt.savefig(prefix + "_loss.png"); plt.close()

            # accuracy
            plt.figure(figsize=(8,5))
            plt.plot(history.history["accuracy"], label="Treino")
            plt.plot(history.history["val_accuracy"], label="Validação")
            plt.title(f"Acurácia por época - act={activation} lr={lr}")
            plt.xlabel("Época"); plt.ylabel("Acurácia")
            plt.legend(); plt.grid(True)
            plt.savefig(prefix + "_acc.png"); plt.close()

            # confusion matrix
            plt.figure(figsize=(5,4))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f"Matriz de Confusão - act={activation} lr={lr}")
            plt.colorbar()
            ticks = np.arange(2)
            plt.xticks(ticks, ["Pontual(0)", "Atrasado(1)"])
            plt.yticks(ticks, ["Pontual(0)", "Atrasado(1)"])
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.ylabel("Classe verdadeira"); plt.xlabel("Classe prevista")
            plt.savefig(prefix + "_confusion.png"); plt.close()

    
            with open(prefix + "_report.txt", "w") as f:
                f.write(f"Activation: {activation}\nLearning rate: {lr}\n")
                f.write(f"Test loss: {test_loss:.6f}\nTest accuracy: {test_acc:.6f}\n\n")
                f.write("Classification report:\n")
                f.write(cr)
                f.write("\nConfusion matrix:\n")
                f.write(np.array2string(cm))

            summary_rows.append({
                "activation": activation,
                "lr": lr,
                "test_loss": float(test_loss),
                "test_accuracy": float(test_acc),
                "train_samples": X_train.shape[0],
                "test_samples": X_test.shape[0],
                "n_features": X_train.shape[1]
            })
            print(f"Resultados salvos: {prefix}_*.png e {prefix}_report.txt")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_results.csv"), index=False)
    print("Resumo final salvo em outputs/summary_results.csv")

if __name__ == "__main__":
    df = load_data()
    run_experiment(df)
