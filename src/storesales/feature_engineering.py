import pandas as pd
import numpy as np


def create_volume_tiers(
    df: pd.DataFrame,
    tier_feats: list,
    target_col: str,
    cutoff_date: str,
    n_tiers: int = 5,
) -> pd.DataFrame:

    df_out = df.copy()
    historical_df = df_out[df_out["date"] < cutoff_date]

    for col in tier_feats:
        total_volume = historical_df.groupby(col)[target_col].sum()
        volume_log = np.log1p(total_volume)
        tiers = pd.cut(volume_log, bins=n_tiers, labels=range(1, n_tiers + 1))

        tier_mapping = tiers.to_dict()
        new_col = f"{col}_volume_tier"
        df_out[new_col] = df_out[col].map(tier_mapping).fillna(1).astype(int)

    return df_out
