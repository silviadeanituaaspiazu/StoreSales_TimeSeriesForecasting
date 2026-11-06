from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.linear_model import LinearRegression
import numpy as np


def get_linear_regression_pipeline(skewed_feats, normal_feats, cat_feats):

    log_transformer = FunctionTransformer(np.log1p, validate=True)

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
