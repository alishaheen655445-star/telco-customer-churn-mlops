import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.data.preprocessing import load_data, clean_data
from src.features.data_split import split_data
from src.features.preprocessing_pipeline import build_preprocessing_pipeline


def evaluate_model(name, model, X_train, X_test, y_train, y_test, preprocessor):

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    with mlflow.start_run(run_name=name):

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0

        mlflow.log_param("model", name)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc)

        mlflow.sklearn.log_model(pipeline, name="model")

        print(f"\n{name} -> F1: {f1:.4f}")

        return f1, pipeline


def run_training():

    print("Loading data...")
    df = load_data()
    df = clean_data(df)

    print("\n========== Dataset Shape After Cleaning ==========")
    print(df.shape)

    X_train, X_test, y_train, y_test = split_data(df)

    print("\nTraining started...")

    # ✅ IMPORTANT FIX HERE
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Telco_Churn_Project")

    preprocessor = build_preprocessing_pipeline(X_train)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "DecisionTree": DecisionTreeClassifier()
    }

    best_f1 = 0
    best_model = None
    best_name = ""

    for name, model in models.items():

        print(f"\nTraining {name}...")

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

    print("\n================ BEST MODEL ================")
    print(f"Model: {best_name}")
    print(f"F1 Score: {best_f1:.4f}")

    with mlflow.start_run(run_name="BEST_MODEL_FINAL"):
        mlflow.sklearn.log_model(best_model, name="best_model")
        mlflow.log_metric("best_f1", best_f1)


if __name__ == "__main__":
    run_training()