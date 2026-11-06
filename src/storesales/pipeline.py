import os
import json
import mlflow
import pandas as pd

from src.storesales.training import train_predict_evaluate
from src.storesales.visualization import plot_predictions
from src.storesales.models import get_linear_regression_pipeline

MODEL_REGISTRY = {
    "linear_regression": get_linear_regression_pipeline,
    # "xgboost": get_xgboost_pipeline,
    # "random_forest": get_rf_pipeline,
}


def run_experiment(
    model_name: str,
    skewed_feats: list,
    normal_feats: list,
    cat_feats: list,
    target: str,
    cols_to_keep: list,
    cutoffs: list,
    horizon: int,
    issue_name: str,
    version_name: str,
    df_path: str = "../data/processed/df_train.parquet",
    to_plot_family_name: str = "AUTOMOTIVE",
    to_plot_store_nbr_name: int = 20,
    to_plot_cutoff: str = "2017-07-15",
):
    df = pd.read_parquet(df_path)
    features = skewed_feats + normal_feats + cat_feats
    selected_model = MODEL_REGISTRY[model_name]
    run_name = f"{model_name}_{issue_name}_{version_name}"

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    db_path = os.path.join(project_root, "mlflow.db")
    artifacts_dir = os.path.join(project_root, "artifacts")
    mlruns_dir = os.path.join(project_root, "mlruns")

    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    client = mlflow.tracking.MlflowClient()
    experiment_name = "Store_Sales_Forecasting"
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        client.create_experiment(
            experiment_name, artifact_location=f"file://{mlruns_dir}"
        )

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):

        dataset_name = os.path.splitext(os.path.basename(df_path))[0]
        dataset = mlflow.data.from_pandas(df, source=df_path, name=dataset_name)
        mlflow.log_input(dataset, context="training")

        mlflow.set_tag("issue", issue_name)
        mlflow.set_tag("version", version_name)

        mlflow.log_param("model_type", model_name)
        mlflow.log_param("horizon_days", horizon)
        mlflow.log_param("total_features", len(features))
        mlflow.log_param("cutoff", [str(c.date()) for c in cutoffs])

        os.makedirs(artifacts_dir, exist_ok=True)
        features_path = os.path.join(artifacts_dir, "features_list.json")

        with open(features_path, "w") as f:
            json.dump({"features": features}, f, indent=4)

        mlflow.log_artifact(features_path, "metadata")

        df_results = train_predict_evaluate(
            df,
            selected_model,
            cutoffs,
            horizon,
            skewed_feats,
            normal_feats,
            cat_feats,
            target,
            cols_to_keep,
        )

        plot_path = plot_predictions(
            df,
            df_results,
            run_name,
            to_plot_family_name,
            to_plot_store_nbr_name,
            to_plot_cutoff,
        )
        mlflow.log_artifact(plot_path, "visualizations")

    return df_results
