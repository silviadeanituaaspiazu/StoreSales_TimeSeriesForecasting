from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_log_error,
    root_mean_squared_error,
)


def accuracy_evaluation(y_true, y_pred, cutoff):
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    rmse = root_mean_squared_error(y_true=y_true, y_pred=y_pred)
    rmsle = root_mean_squared_log_error(y_true=y_true, y_pred=y_pred)

    print(
        f"Cutoff: {cutoff.date()}, MAE: {mae:.2f}, RMSE:{rmse:.2f}, RMSLE:{rmsle:.2f}"
    )

    return {"mae": mae, "rmse": rmse, "rmsle": rmsle}
