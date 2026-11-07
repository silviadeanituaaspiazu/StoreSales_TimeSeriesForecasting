from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.linear_model import LinearRegression
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def get_linear_regression_pipeline(skewed_feats, normal_feats, cat_feats):

    log_transformer = FunctionTransformer(
        np.log1p, validate=True, feature_names_out="one-to-one"
    )

    skewed_pipeline = Pipeline(
        steps=[("log", log_transformer), ("scaler", StandardScaler())]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_skewed", skewed_pipeline, skewed_feats),
            ("num_normal", StandardScaler(), normal_feats),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
        ]
    )

    model = Pipeline(
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
    return model


def get_xgboost_model(skewed_feats, normal_feats, cat_feats):
    """The parameters are not used but they are necessary for the project structure"""

    preprocessor = FunctionTransformer(feature_names_out="one-to-one")
    preprocessor.set_output(transform="pandas")

    base_xgb = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        verbosity=0,
        enable_categorical=True,
        tree_method="hist",
    )

    model = TransformedTargetRegressor(
        regressor=base_xgb, func=np.log1p, inverse_func=np.expm1
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline


def get_lightgbm_model(skewed_feats, normal_feats, cat_feats):
    """The parameters are not used but they are necessary for the project structure"""

    preprocessor = FunctionTransformer(feature_names_out="one-to-one")
    preprocessor.set_output(transform="pandas")

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
