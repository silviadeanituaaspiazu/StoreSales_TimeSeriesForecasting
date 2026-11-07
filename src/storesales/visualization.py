import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import root_mean_squared_log_error


def plot_rmsle_by_step(df_results, root_artifacts):
    plot_path = os.path.join(root_artifacts, "plot_rmsle_by_step_temp.png")

    steps = []
    rmsle_values = []

    for step, group in df_results.groupby("forecast_step"):
        rmsle = root_mean_squared_log_error(group["y_true"], group["y_pred"])
        steps.append(step)
        rmsle_values.append(rmsle)

    plt.figure(figsize=(10, 6))
    plt.plot(
        steps,
        rmsle_values,
        marker="o",
        linestyle="-",
        color="mediumorchid",
        linewidth=2.5,
        markersize=8,
        label="RMSLE",
    )
    plt.xlabel("Forecast Step (Days)", fontsize=12)
    plt.ylabel("RMSLE", fontsize=12)
    plt.title(
        "RMSLE Degradation by Forecast Step", fontsize=14, fontweight="bold", pad=15
    )
    plt.xticks(np.array(steps).astype(int))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_path


def plot_actual_vs_predicted(y_true, y_pred, root_artifacts):
    plot_path = os.path.join(root_artifacts, "plot_actual_vs_predicted_temp.png")

    plt.figure(figsize=(10, 6))
    plt.scatter(
        y_true,
        y_pred,
        alpha=0.4,
        color="dodgerblue",
        edgecolors="white",
        linewidth=0.8,
        label="Predictions",
    )

    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot(
        [0, max_val],
        [0, max_val],
        color="crimson",
        linestyle="--",
        linewidth=2.5,
        label="Perfect Prediction (y = x)",
    )
    plt.xlabel("Actual Sales", fontsize=12)
    plt.ylabel("Predicted Sales", fontsize=12)
    plt.title("Actual vs. Predicted Sales", fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="upper left", frameon=True, shadow=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(0, max_val * 1.05)
    plt.ylim(0, max_val * 1.05)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_path


def plot_feature_importance(model, root_artifacts):
    plot_path = os.path.join(root_artifacts, "plot_importances_temp.png")

    final_model = model.named_steps["model"]
    col_names = model.named_steps["preprocessor"].get_feature_names_out()

    if hasattr(final_model, "feature_importances_"):
        importance = final_model.feature_importances_
    elif hasattr(final_model, "coef_"):
        importance = final_model.coef_
    else:
        return None

    df_importance = (
        pd.DataFrame({"Variable": col_names, "Importance": importance})
        .sort_values(by="Importance", ascending=True)
        .tail(20)
    )

    plt.figure(figsize=(10, 7))
    plt.barh(
        df_importance["Variable"],
        df_importance["Importance"],
        color="#2c7bb6",
        height=0.6,
    )

    plt.xlabel("Importance", fontsize=12, labelpad=10)
    plt.title("Top 20 Feature Importance", fontsize=14, fontweight="bold", pad=15)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_path


def plot_predictions(
    df: pd.DataFrame,
    df_results: pd.DataFrame,
    family_name: str,
    store_nbr_name: int,
    cutoff: str,
    root_artifacts: str,
):
    plot_path = os.path.join(root_artifacts, "plot_predictions_temp.png")
    cutoff = pd.to_datetime(cutoff)

    df_results_filtered = df_results[
        (df_results["family"] == family_name)
        & (df_results["store_nbr"] == store_nbr_name)
        & (df_results["cutoff"] == cutoff)
    ]
    df_filtered = df[
        (df["family"] == family_name)
        & (df["store_nbr"] == store_nbr_name)
        & (df["date"] <= cutoff)
        & (df["date"] > (cutoff - pd.Timedelta(days=90)))
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(16, 5))

    plt.plot(
        df_filtered["date"],
        df_filtered["sales"],
        color="#8c8c8c",
        linewidth=1,
        label="Historical Sales",
    )
    plt.plot(
        df_results_filtered["date"],
        df_results_filtered["y_true"],
        color="#005b96",
        linewidth=1,
        marker="o",
        markersize=3,
        label="Actual Sales",
    )
    plt.plot(
        df_results_filtered["date"],
        df_results_filtered["y_pred"],
        color="#e34a33",
        linewidth=1,
        linestyle="--",
        marker="X",
        markersize=3,
        label="Model Prediction",
    )
    plt.axvline(
        x=cutoff,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Cutoff Date",
    )
    plt.title(
        f"Sales Prediction: {family_name.title()} | Store {store_nbr_name}",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Date", fontsize=12, labelpad=10)
    plt.ylabel("Units Sold", fontsize=12, labelpad=10)
    plt.legend(loc="upper left", fontsize=10, frameon=True, shadow=True, borderpad=1)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_path


def generate_and_log_plots(df, df_results, model, config):
    root_artifacts = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../artifacts/plots")
    )
    os.makedirs(root_artifacts, exist_ok=True)

    path_predictions = plot_predictions(
        df,
        df_results,
        config["family_name"],
        config["store_nbr"],
        config["cutoff"],
        root_artifacts,
    )
    mlflow.log_artifact(path_predictions, "visualizations")

    path_importance = plot_feature_importance(model, root_artifacts)
    if path_importance:
        mlflow.log_artifact(path_importance, "visualizations")

    path_actual_pred = plot_actual_vs_predicted(
        df_results["y_true"], df_results["y_pred"], root_artifacts
    )
    mlflow.log_artifact(path_actual_pred, "visualizations")

    path_rmsle = plot_rmsle_by_step(df_results, root_artifacts)
    mlflow.log_artifact(path_rmsle, "visualizations")
