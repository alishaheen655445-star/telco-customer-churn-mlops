import mlflow
import pandas as pd


def select_best_model():

    # 👇 نفس قاعدة البيانات
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    experiment_name = "Telco_Churn_Project"

    # =========================
    # Get experiment
    # =========================
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        print("Experiment not found")
        return

    experiment_id = experiment.experiment_id

    # =========================
    # Get all runs
    # =========================
    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    if runs.empty:
        print("No runs found")
        return

    # =========================
    # Choose best model
    # (based on ROC AUC)
    # =========================
    best_run = runs.sort_values("metrics.roc_auc", ascending=False).iloc[0]

    best_run_id = best_run.run_id
    best_score = best_run["metrics.roc_auc"]

    print("\n========== BEST MODEL ==========")
    print(f"Run ID   : {best_run_id}")
    print(f"ROC AUC  : {best_score}")

    # =========================
    # Register best model
    # =========================
    model_uri = f"runs:/{best_run_id}/model"

    mlflow.register_model(
        model_uri=model_uri,
        name="TelcoChurnBestModel"
    )

    print("\nModel registered successfully ✔")


if __name__ == "__main__":
    select_best_model()