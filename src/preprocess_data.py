import pandas as pd
import os

def preprocess_data():
    # Caminhos
    raw_path = "data/ontime_reporting.csv"  # arquivo correto
    processed_dir = "src/processed"          # pasta para salvar processado
    processed_path = f"{processed_dir}/processed_flights.csv"

    # Criar pasta processed se não existir
    os.makedirs(processed_dir, exist_ok=True)

    print(f"🔄 Carregando o dataset bruto: {raw_path}")
    df = pd.read_csv(raw_path)

    print("🔧 Limpando dados...")

    # Remover linhas com valores faltando nos principais atributos
    df = df.dropna(subset=["DepDelay", "ArrDelay", "Distance", "AirTime"])

    # Criando uma coluna binária de atraso (0 = no horário, 1 = atrasou mais de 15 min)
    df["Delayed"] = (df["ArrDelay"] > 15).astype(int)

    # Selecionando atributos importantes
    df_processed = df[[
        "DepDelay",
        "Distance",
        "AirTime",
        "ArrDelay",
        "Delayed"
    ]]

    print("💾 Salvando dataset processado...")
    df_processed.to_csv(processed_path, index=False)

    print("✅ Arquivo processado criado com sucesso!")
    print(f"📁 Local: {processed_path}")


if __name__ == "__main__":
    preprocess_data()
