from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer,
    TargetEncoder,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.linear_model import LinearRegression
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def get_linear_regression_pipeline(
    skewed_feats, normal_feats, tier_feats, cat_feats, target_encoder: bool = False
):
    """Gets the linear model. The target encoder function is not available"""

    if target_encoder:
        raise NotImplementedError(
            "Target Encoding is not supported for Linear Regression in this pipeline. "
            "Please run with target_encoder=False."
        )

    tier_cols = [f"{col}_volume_tier" for col in tier_feats] if tier_feats else []
    all_normal_feats = normal_feats + tier_cols

    log_transformer = FunctionTransformer(
        np.log1p, validate=True, feature_names_out="one-to-one"
    )

    skewed_pipeline = Pipeline(
        steps=[("log", log_transformer), ("scaler", StandardScaler())]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_skewed", skewed_pipeline, skewed_feats),
            ("num_normal", StandardScaler(), all_normal_feats),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                TransformedTargetRegressor(
                    regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1
                ),
            ),
        ]
    )
    return pipeline


def _get_tree_preprocessor(
    skewed_feats, normal_feats, tier_feats, cat_feats, target_encoder: bool
):
    """Helper function to build the preprocessor for tree-based models."""
    tier_cols = [f"{col}_volume_tier" for col in tier_feats] if tier_feats else []
    all_num_feats = normal_feats + skewed_feats + tier_cols

    if target_encoder:
        cat_transformer = TargetEncoder(target_type="continuous", random_state=42)
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", cat_transformer, cat_feats),
                ("num", "passthrough", all_num_feats),
            ],
            remainder="drop",
        )
    else:
        preprocessor = FunctionTransformer(feature_names_out="one-to-one")
        preprocessor.set_output(transform="pandas")

    return preprocessor


def get_xgboost_model(
    skewed_feats, normal_feats, tier_feats, cat_feats, target_encoder: bool = False
):
    preprocessor = _get_tree_preprocessor(
        skewed_feats, normal_feats, tier_feats, cat_feats, target_encoder
    )

    base_xgb = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        verbosity=0,
        enable_categorical=True,
        tree_method="hist",
        # objective="reg:tweedie"
    )

    model = TransformedTargetRegressor(
        regressor=base_xgb, func=np.log1p, inverse_func=np.expm1
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline


def get_lightgbm_model(
    skewed_feats, normal_feats, tier_feats, cat_feats, target_encoder: bool = False
):
    preprocessor = _get_tree_preprocessor(
        skewed_feats, normal_feats, tier_feats, cat_feats, target_encoder
    )

    base_lgbm = LGBMRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
    )

    model = TransformedTargetRegressor(
        regressor=base_lgbm, func=np.log1p, inverse_func=np.expm1
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline
