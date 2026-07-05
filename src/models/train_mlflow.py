import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.features.preprocessing_pipeline import build_preprocessing_pipeline
from src.features.data_split import split_data
from src.data.preprocessing import load_data, clean_data


def run_training():

    # =========================
    # 1. Load Dataset
    # =========================
    print("\n========== Loading Data ==========")

    df = load_data()          # Load raw dataset
    df = clean_data(df)       # Clean missing values / fix data issues

    print("\nDataset Shape:", df.shape)

    # =========================
    # 2. Split Dataset
    # =========================
    # X_train → training features
    # X_test  → testing features
    # y_train → training labels
    # y_test  → testing labels
    X_train, X_test, y_train, y_test = split_data(df)

    # =========================
    # 3. MLflow Setup
    # =========================
    # Store experiments in SQLite database (NOT file-based mlruns)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Create or reuse experiment
    mlflow.set_experiment("Telco_Churn_Project")

    # =========================
    # 4. Define Models
    # =========================
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "DecisionTree": DecisionTreeClassifier()
    }

    best_model = None
    best_score = 0
    best_name = ""

    # =========================
    # 5. Preprocessing Pipeline
    # =========================
    # Handles encoding + scaling automatically
    preprocessor = build_preprocessing_pipeline(X_train)

    # =========================
    # 6. Train Models
    # =========================
    for name, model in models.items():

        print(f"\n🚀 Training model: {name}")

        with mlflow.start_run(run_name=name):

            # Build full ML pipeline (preprocessing + model)
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            # Train model
            pipeline.fit(X_train, y_train)

            # Predictions
            y_pred = pipeline.predict(X_test)

            # Probability predictions (if supported)
            y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            # =========================
            # 7. Evaluation Metrics
            # =========================
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0

            # =========================
            # 8. Log to MLflow
            # =========================
            mlflow.log_param("model_name", name)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            mlflow.sklearn.log_model(pipeline, name="model")

            print(f"✔ {name} Accuracy: {accuracy:.4f}")

            # =========================
            # 9. Track Best Model
            # =========================
            if accuracy > best_score:
                best_score = accuracy
                best_model = pipeline
                best_name = name

    # =========================
    # 10. Save Best Model
    # =========================
    print("\n🏆 Best Model:", best_name)
    print("📊 Best Accuracy:", best_score)

    with mlflow.start_run(run_name="BEST_MODEL_FINAL"):

        mlflow.log_param("best_model", best_name)
        mlflow.log_metric("best_accuracy", best_score)

        mlflow.sklearn.log_model(best_model, name="best_model")

    print("\n🎯 Best model successfully saved in MLflow!")


if __name__ == "__main__":
    run_training()