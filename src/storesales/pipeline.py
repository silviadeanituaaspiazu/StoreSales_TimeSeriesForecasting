import os
import mlflow
import pandas as pd

from storesales.training import train_predict_evaluate
from storesales.visualization import generate_and_log_plots
from storesales.models import (
    get_linear_regression_pipeline,
    get_xgboost_model,
    get_lightgbm_model,
)

MODEL_REGISTRY = {
    "linear_regression": get_linear_regression_pipeline,
    "xgboost": get_xgboost_model,
    "lightgbm": get_lightgbm_model,
}


def run_experiment(
    model_name: str,
    skewed_feats: list,
    normal_feats: list,
    tier_feats: list,
    cat_feats: list,
    target_encoder: bool,
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

    for cat in cat_feats:
        df[cat] = df[cat].astype("category")

    tier_cols = [f"{col}_volume_tier" for col in tier_feats] if tier_feats else []
    features = skewed_feats + normal_feats + cat_feats + tier_cols

    selected_model = MODEL_REGISTRY[model_name]
    model_instance = selected_model(
        skewed_feats, normal_feats, tier_feats, cat_feats, target_encoder
    )

    run_name = f"{model_name}_{issue_name}_{version_name}"

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    artifacts_dir = os.path.join(project_root, "artifacts")
    results_path = os.path.join(artifacts_dir, f"predictions/pred_{run_name}.parquet")

    # Rutas absolutas para MLflow
    db_path = os.path.join(project_root, "mlflow.db")
    mlruns_dir = os.path.join(project_root, "mlruns")

    # 1. Conectar a la base de datos principal
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    experiment_name = "Store_Sales_Forecasting"
    client = mlflow.tracking.MlflowClient()

    # 2. Replicamos tu lógica infalible para fijar la ruta de artifacts en la raíz
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(
            name=experiment_name, artifact_location=f"file://{mlruns_dir}"
        )

    # 3. Activamos el experimento
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):

        dataset_name = os.path.splitext(os.path.basename(df_path))[0]
        dataset = mlflow.data.from_pandas(df, source=df_path, name=dataset_name)
        mlflow.log_input(dataset, context="training")

        mlflow.set_tags({"issue": issue_name, "version": version_name, "data": "valid"})

        mlflow.log_params(
            {
                "model_type": model_name,
                "horizon_days": horizon,
                "total_features": len(features),
                "target_encoder": target_encoder,
                "cutoff": [str(c)[:10] for c in cutoffs],
            }
        )

        algo_wrapper = model_instance.named_steps["model"]
        base_algo = algo_wrapper.regressor
        mlflow.log_params(base_algo.get_params())

        mlflow.log_dict({"features": features}, "metadata/features_list.json")

        df_results, trained_models = train_predict_evaluate(
            df,
            selected_model,
            cutoffs,
            horizon,
            skewed_feats,
            normal_feats,
            tier_feats,
            cat_feats,
            target_encoder,
            target,
            cols_to_keep,
        )

        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        df_results.to_parquet(results_path)
        mlflow.log_artifact(results_path, "predictions")

        plot_config = {
            "family_name": to_plot_family_name,
            "store_nbr": to_plot_store_nbr_name,
            "cutoff": to_plot_cutoff,
        }

        last_trained_model = list(trained_models.values())[-1]
        generate_and_log_plots(
            df=df, df_results=df_results, model=last_trained_model, config=plot_config
        )

    return df_results, trained_models
