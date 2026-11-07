import mlflow.sklearn
import pandas as pd
import numpy as np

from src.storesales.metrics import accuracy_evaluation


def train_predict_evaluate(
    df: pd.DataFrame,
    model_builder_fn,
    cutoffs: list,
    horizon: int,
    skewed_feats: list,
    normal_feats: list,
    cat_feats: list,
    target: str,
    cols_to_keep: list,
):

    horizon_days = pd.Timedelta(days=horizon)

    results_list = []
    metrics_list = []

    features = skewed_feats + normal_feats + cat_feats

    for i, cutoff in enumerate(cutoffs):
        model = model_builder_fn(skewed_feats, normal_feats, cat_feats)

        train_df = df[df["date"] < cutoff]
        test_df = df[(df["date"] >= cutoff) & (df["date"] <= (cutoff + horizon_days))]

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

        mask = res["is_int_sales"] == True
        res.loc[mask, "y_pred"] = res.loc[mask, "y_pred"].round()

        fold_metrics = accuracy_evaluation(y_test, y_pred, cutoff)
        metrics_list.append(fold_metrics)
        results_list.append(res)

        mlflow.log_metric(f"rmsle_fold_{i+1}", fold_metrics["rmsle"])

    avg_rmsle = np.mean([m["rmsle"] for m in metrics_list])
    print(f"GLOBAL RMSLE: {avg_rmsle:.4f}")

    mlflow.log_metric("global_rmsle", avg_rmsle)
    mlflow.sklearn.log_model(
        model, "pipeline_model", serialization_format="cloudpickle"
    )

    df_results = pd.concat(results_list)
    return df_results, model
