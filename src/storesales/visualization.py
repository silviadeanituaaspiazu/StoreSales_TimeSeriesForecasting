import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_predictions(
    df: pd.DataFrame,
    df_results: pd.DataFrame,
    run_name: str,
    family_name: str = "AUTOMOTIVE",
    store_nbr_name: int = 20,
    cutoff: str = "2017-07-15",
):

    root_artifacts = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../artifacts")
    )
    os.makedirs(root_artifacts, exist_ok=True)

    plot_filename = f"plot_predictions_{run_name}_{family_name}_{store_nbr_name}.png"
    plot_path = os.path.join(root_artifacts, plot_filename)

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
    plt.show()
    plt.close()

    return plot_path
