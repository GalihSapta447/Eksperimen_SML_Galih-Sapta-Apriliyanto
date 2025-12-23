import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "..", "Exam_Score_Prediction.csv")

    print(f"Mencoba membaca file dari: {file_path}")

    try:
        # Load raw dataset
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {file_path}")
        print(f"Isi folder saat ini: {os.listdir(os.path.dirname(file_path))}")
        return

    # Separate features and target
    X = df.drop("exam_score", axis=1)
    y = df["exam_score"]

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

    # One-Hot Encoding for categorical features
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Standardization for numerical features
    scaler = StandardScaler()
    X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

    # Combine features and target
    df_preprocessed = pd.concat([X_encoded, y], axis=1)

    # Simpan hasil preprocessing ke folder yang sama dengan file input
    output_path = os.path.join(current_dir, "..", "exam_score_preprocessed.csv")
    df_preprocessed.to_csv(output_path, index=False)

    print(f"Preprocessing selesai. Dataset disimpan di: {output_path}")

if __name__ == "__main__":
    main()
