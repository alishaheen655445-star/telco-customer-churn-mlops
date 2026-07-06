import mlflow
import mlflow.sklearn
import joblib

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.data.preprocessing import load_data, clean_data
from src.features.data_split import split_data
from src.features.preprocessing_pipeline import build_preprocessing_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def evaluate_model(name, model, X_train, X_test, y_train, y_test, preprocessor):

    # Build pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Probabilities
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0

    # MLflow logs
    mlflow.log_param("model_name", name)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("roc_auc", roc)

    # Save model inside MLflow
    mlflow.sklearn.log_model(pipeline, name="model")

    print(f"{name} -> F1: {f1:.4f}")

    return f1, pipeline


def run_training():

    print("Loading data...")

    # Load data
    df = load_data()
    df = clean_data(df)

    print("Dataset shape:", df.shape)

    # Split
    X_train, X_test, y_train, y_test = split_data(df)

    # Preprocess
    preprocessor = build_preprocessing_pipeline(X_train)

    # MLflow config (FIXED)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Telco_Churn_Project")

    # Models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "DecisionTree": DecisionTreeClassifier()
    }

    best_f1 = 0
    best_model = None
    best_name = ""

    for name, model in models.items():

        print(f"Training {name}...")

        with mlflow.start_run(run_name=name):

            f1, pipeline = evaluate_model(
                name,
                model,
                X_train,
                X_test,
                y_train,
                y_test,
                preprocessor
            )

            if f1 > best_f1:
                best_f1 = f1
                best_model = pipeline
                best_name = name

    # IMPORTANT: save model for API
    joblib.dump(best_model, "model.pkl")

    print("\nBEST MODEL:", best_name)
    print("BEST F1:", best_f1)
    print("Saved: model.pkl")


if __name__ == "__main__":
    run_training()