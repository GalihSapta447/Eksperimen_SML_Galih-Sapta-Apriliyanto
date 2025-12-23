import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_path, "..", "Exam_Score_Prediction.csv")
    output_path = os.path.join(base_path, "exam_score_preprocessed.csv")

    print(f"Membaca dataset dari: {input_path}")

    # Load dataset
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File {input_path} tidak ditemukan!")
    
    df = pd.read_csv(input_path)

    # Preprocessing (Sesuai eksperimen notebook)
    X = df.drop("exam_score", axis=1)
    y = df["exam_score"]

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    scaler = StandardScaler()
    X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

    df_preprocessed = pd.concat([X_encoded, y], axis=1)

    # Simpan hasil
    df_preprocessed.to_csv(output_path, index=False)
    print(f"Preprocessing selesai! File disimpan di: {output_path}")

if __name__ == "__main__":
    main()
