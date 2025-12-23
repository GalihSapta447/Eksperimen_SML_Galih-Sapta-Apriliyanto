import pandas as pd
from sklearn.preprocessing import StandardScaler


def main():
    # Load raw dataset
    df = pd.read_csv("Exam_Score_Prediction.csv")

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

    # Save preprocessed dataset
    df_preprocessed.to_csv(
        "exam_score_preprocessed.csv",
        index=False
    )

    print("Preprocessing selesai. Dataset disimpan.")


if __name__ == "__main__":
    main()
