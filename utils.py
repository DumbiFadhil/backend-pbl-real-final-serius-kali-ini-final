import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_error(y_true, y_pred):
    """
    Calculate regression error metrics:
    - RMSE
    - Correlation
    - MAE
    - RAE (Relative Absolute Error)
    - RRSE (Root Relative Squared Error)
    - MAPE (Mean Absolute Percentage Error)
    - R2 (R-squared)
    """
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    mae = mean_absolute_error(y_true, y_pred)
    rae = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))
    rrse = np.sqrt(np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return rmse, corr, mae, rae, rrse, mape, r2

def save_latest_model_id(model_id, path="latest_model_id.txt"):
    with open(path, "w") as f:
        f.write(model_id)

def load_latest_model_id(path="latest_model_id.txt"):
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception:
        return None