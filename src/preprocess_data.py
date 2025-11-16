# src/preprocess_data.py
import os
import pandas as pd

RAW_PATH = os.path.join("data", "raw", "ontime_reporting.csv")
PROCESSED_DIR = os.path.join("data", "processed")
PROCESSED_PATH = os.path.join(PROCESSED_DIR, "processed_flights.csv")

def main(raw_path=RAW_PATH, processed_path=PROCESSED_PATH):
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Arquivo bruto não encontrado: {raw_path}")

    print("Carregando CSV bruto:", raw_path)
    df = pd.read_csv(raw_path)

  
    required = {"FL_DATE","OP_UNIQUE_CARRIER","ORIGIN","DEST","CRS_DEP_TIME","DEP_DELAY","ARR_DELAY","DISTANCE"}
    missing = required - set(df.columns)
    if missing:
        print("Atenção: colunas esperadas faltando no CSV bruto:", missing)
        
    rename_map = {
        "OP_UNIQUE_CARRIER": "UniqueCarrier",
        "CRS_DEP_TIME": "CRSDepTime",
        "DEP_DELAY": "DepDelay",
        "ARR_DELAY": "ArrDelay",
        "DISTANCE": "Distance",
        "FL_DATE": "FlightDate",
        "ORIGIN": "Origin",
        "DEST": "Dest"
    }
    df = df.rename(columns=rename_map)

  
    if "CRSDepTime" in df.columns:
        df["CRSDepTime"] = pd.to_numeric(df["CRSDepTime"], errors="coerce")

    
    for col in ["DepDelay", "ArrDelay", "Distance"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    
    keep_cols = [c for c in ["DepDelay","ArrDelay","Distance"] if c in df.columns]
    df = df.dropna(subset=keep_cols)

   
    if "ArrDelay" not in df.columns:
        raise KeyError("Coluna 'ArrDelay' não encontrada no CSV após renomear.")
    df["Delayed"] = (df["ArrDelay"] > 15).astype(int)


    features = []
    for c in ["FlightDate", "CRSDepTime", "DepDelay", "Distance", "UniqueCarrier", "Origin", "Dest"]:
        if c in df.columns:
            features.append(c)

    df_processed = df[features + ["Delayed"]].copy()

    
    if "FlightDate" in df_processed.columns:
        df_processed["FlightDate"] = pd.to_datetime(df_processed["FlightDate"], errors="coerce")
        df_processed["Year"] = df_processed["FlightDate"].dt.year
        df_processed["Month"] = df_processed["FlightDate"].dt.month
        df_processed["Day"] = df_processed["FlightDate"].dt.day
        df_processed["DayOfWeek"] = df_processed["FlightDate"].dt.dayofweek
        df_processed = df_processed.drop(columns=["FlightDate"])

    
    cat_cols = [c for c in ["UniqueCarrier", "Origin", "Dest"] if c in df_processed.columns]
    if cat_cols:
        df_processed = pd.get_dummies(df_processed, columns=cat_cols, drop_first=True)

 
    cols = [c for c in df_processed.columns if c != "Delayed"] + ["Delayed"]
    df_processed = df_processed[cols]

 
    df_processed.to_csv(processed_path, index=False)
    print("Arquivo processado salvo em:", processed_path)
    print("Registros finais:", len(df_processed))
    print("Atributos finais:", df_processed.shape[1])

if __name__ == "__main__":
    main()
