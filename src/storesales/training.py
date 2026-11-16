import mlflow.sklearn
import pandas as pd
import numpy as np
import mlflow

from storesales.metrics import accuracy_evaluation
from storesales.feature_engineering import create_volume_tiers

N_TIERS = 5


def train_predict_evaluate(
    df: pd.DataFrame,
    model_builder_fn,
    cutoffs: list,
    horizon: int,
    skewed_feats: list,
    normal_feats: list,
    tier_feats: list,
    cat_feats: list,
    target_encoder: bool,
    target: str,
    cols_to_keep: list,
):

    horizon_days = pd.Timedelta(days=horizon)

    results_list = []
    metrics_list = []

    models_dict = {}

    tier_features = [f"{col}_volume_tier" for col in tier_feats]
    features = skewed_feats + normal_feats + cat_feats + tier_features

    for i, cutoff in enumerate(cutoffs):

        if tier_feats:
            df = create_volume_tiers(
                df=df,
                tier_feats=tier_feats,
                target_col=target,
                cutoff_date=cutoff,
                n_tiers=N_TIERS,
            )

        model = model_builder_fn(skewed_feats, normal_feats, tier_feats, cat_feats, target_encoder)

        train_df = df[df["date"] < cutoff]
        test_df = df[(df["date"] >= cutoff) & (df["date"] < (cutoff + horizon_days))]

        X_train, X_test = train_df[features], test_df[features]
        y_train, y_test = train_df[target], test_df[target]

        model.fit(X_train, y_train)
        y_pred = np.clip(model.predict(X_test), a_min=0, a_max=None)

        res = X_test.copy()

        for col in cols_to_keep:
            res[col] = test_df[col]

        res["cutoff"] = cutoff
        res["y_true"] = y_test
        res["y_pred"] = y_pred
        res["forecast_step"] = (res["date"] - res["cutoff"]).dt.days + 1

        mask = res["is_int_sales"] == True
        res.loc[mask, "y_pred"] = res.loc[mask, "y_pred"].round()

        fold_metrics = accuracy_evaluation(y_test, y_pred, cutoff)
        metrics_list.append(fold_metrics)
        results_list.append(res)

        mlflow.log_metric(f"rmsle_fold_{i+1}", fold_metrics["rmsle"])
        mlflow.log_metric(f"bias_fold_{i+1}", fold_metrics["bias"])

        models_dict[cutoff] = model
        mlflow.sklearn.log_model(
            model, f"pipeline_model_fold_{i+1}", serialization_format="cloudpickle"
        )

    df_results = pd.concat(results_list)

    avg_rmsle = np.mean([m["rmsle"] for m in metrics_list])
    global_bias = (df_results["y_pred"] - df_results["y_true"]).mean()
    print(f"GLOBAL RMSLE: {avg_rmsle:.4f} GLOBAL BIAS:  {global_bias:.4f}")

    mlflow.log_metric("global_rmsle", avg_rmsle)
    mlflow.log_metric("global_bias", global_bias)
    mlflow.sklearn.log_model(
        model, "pipeline_model", serialization_format="cloudpickle"
    )

    return df_results, models_dict
