from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from src.data.preprocessing import load_data, clean_data
from src.features.data_split import split_data
from src.features.preprocessing_pipeline import build_preprocessing_pipeline


def run_training():

    print("\n========== LOADING DATA ==========")
    df = load_data()
    df = clean_data(df)

    print("\n========== SPLITTING DATA ==========")
    X_train, X_test, y_train, y_test = split_data(df)

    print("\n========== BUILDING PIPELINE ==========")

    preprocessor = build_preprocessing_pipeline(X_train)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print("\n========== TRAINING MODEL ==========")

    pipeline.fit(X_train, y_train)

    print("\n========== EVALUATION ==========")

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")

    print("\n========== DONE ==========")


if __name__ == "__main__":
    run_training()