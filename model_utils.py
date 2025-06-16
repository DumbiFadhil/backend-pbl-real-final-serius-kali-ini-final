import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils import compute_error

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}

def get_model_path(model_dir, model_id):
    return os.path.join(model_dir, f"{model_id}.pkl")

def create_store_family_lag_features(
    df, target_col='sales', lag_days=14, 
    date_col='date', store_col='store_nbr', family_col='family'
):
    df_lag = df.copy()
    df_lag[date_col] = pd.to_datetime(df_lag[date_col])
    df_lag = df_lag.sort_values([store_col, family_col, date_col])
    for lag in range(1, lag_days + 1):
        df_lag[f'{target_col}_lag_{lag}'] = df_lag.groupby([store_col, family_col])[target_col].shift(lag)
    for window in [3, 7, 14]:
        if window <= lag_days:
            df_lag[f'{target_col}_rolling_mean_{window}'] = (
                df_lag.groupby([store_col, family_col])[target_col]
                .rolling(window=window, min_periods=1)
                .mean()
                .shift(1)
                .reset_index(level=[0,1], drop=True)
            )
            df_lag[f'{target_col}_rolling_std_{window}'] = (
                df_lag.groupby([store_col, family_col])[target_col]
                .rolling(window=window, min_periods=1)
                .std()
                .shift(1)
                .reset_index(level=[0,1], drop=True)
            )
            df_lag[f'{target_col}_rolling_min_{window}'] = (
                df_lag.groupby([store_col, family_col])[target_col]
                .rolling(window=window, min_periods=1)
                .min()
                .shift(1)
                .reset_index(level=[0,1], drop=True)
            )
            df_lag[f'{target_col}_rolling_max_{window}'] = (
                df_lag.groupby([store_col, family_col])[target_col]
                .rolling(window=window, min_periods=1)
                .max()
                .shift(1)
                .reset_index(level=[0,1], drop=True)
            )
    df_lag[f'{target_col}_diff_1'] = df_lag.groupby([store_col, family_col])[target_col].diff(1)
    df_lag[f'{target_col}_diff_7'] = df_lag.groupby([store_col, family_col])[target_col].diff(7)
    df_lag['day_of_week'] = df_lag[date_col].dt.dayofweek
    df_lag['quarter'] = df_lag[date_col].dt.quarter
    df_lag['is_weekend'] = (df_lag[date_col].dt.dayofweek >= 5).astype(int)
    df_lag['week_of_year'] = df_lag[date_col].dt.isocalendar().week

    # Optionally, add day, month, year to original df for feature info
    df['day'] = pd.to_datetime(df[date_col]).dt.day
    df['month'] = pd.to_datetime(df[date_col]).dt.month
    df['year'] = pd.to_datetime(df[date_col]).dt.year

    df_lag = df_lag.reset_index(drop=True)
    return df_lag

def train_xgboost_model(
    csv_path,
    target_col='sales',
    date_col='date',
    store_col='store_nbr',
    family_col='family',
    lag_days=14,
    test_size=0.2
):
    # Load data
    df = pd.read_csv(csv_path)
    # Preprocess target
    df[target_col] = df[target_col].round().astype(int)
    df = df[df[target_col] > 0]
    df[date_col] = pd.to_datetime(df[date_col])

    # Create lagged features
    df_lag = create_store_family_lag_features(
        df, target_col=target_col, lag_days=lag_days,
        date_col=date_col, store_col=store_col, family_col=family_col
    ).dropna()
    # Features/target split
    drop_cols = ['id', date_col]
    features = [col for col in df_lag.columns if col not in drop_cols + [target_col]]
    X = df_lag[features]
    y = df_lag[target_col]

    categorical_cols = [store_col, family_col]
    numerical_cols = [col for col in features if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }

    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomizedSearchCV(
            xgb, param_distributions, n_iter=10, scoring='neg_mean_squared_error',
            cv=3, verbose=0, n_jobs=-1, random_state=42))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluate
    rmse, corr, mae, rae, rrse, mape, r2 = compute_error(np.ravel(y_test), np.ravel(predictions))
    metrics = dict(
        RMSE=rmse, Correlation=corr, MAE=mae, RAE=rae, RRSE=rrse, MAPE=mape, R2=r2
    )

    # Save last available lagged data for forecasting
    feature_info = dict(
        features=features,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        last_date=str(df_lag[date_col].max())
    )
    # Save last N rows for future forecasting
    last_df_lag = df_lag[df_lag[date_col] == df_lag[date_col].max()]

    return model, metrics, last_df_lag, feature_info

def load_xgboost_pipeline(model_path):
    return joblib.load(model_path)

def forecast_xgboost(
    model_pipeline, last_df_lag, feature_info, n_periods=14,
    target_col='sales', date_col='date', store_col='store_nbr', family_col='family', lag_days=14
):
    # Rolling forecast for the next n_periods (days)
    future_rows = []
    last_df = last_df_lag.copy()
    for i in range(n_periods):
        # Set prediction date to next day
        next_date = last_df[date_col].max() + pd.Timedelta(days=1)
        for idx, row in last_df.groupby([store_col, family_col]).tail(1).iterrows():
            test_row = row.copy()
            test_row[date_col] = next_date
            # Update lag features: shift all lags, insert previous prediction as lag_1
            for lag in reversed(range(2, lag_days + 1)):
                test_row[f'{target_col}_lag_{lag}'] = test_row.get(f'{target_col}_lag_{lag-1}', np.nan)
            test_row[f'{target_col}_lag_1'] = row[target_col]
            # Rolling statistics (simulate: use previous available lags)
            for window in [3, 7, 14]:
                window_vals = [test_row.get(f'{target_col}_lag_{j}', np.nan) for j in range(1, min(window, lag_days)+1)]
                window_vals = [v for v in window_vals if not pd.isnull(v)]
                if window_vals:
                    test_row[f'{target_col}_rolling_mean_{window}'] = np.mean(window_vals)
                    test_row[f'{target_col}_rolling_std_{window}'] = np.std(window_vals)
                    test_row[f'{target_col}_rolling_min_{window}'] = np.min(window_vals)
                    test_row[f'{target_col}_rolling_max_{window}'] = np.max(window_vals)
                else:
                    test_row[f'{target_col}_rolling_mean_{window}'] = np.nan
                    test_row[f'{target_col}_rolling_std_{window}'] = np.nan
                    test_row[f'{target_col}_rolling_min_{window}'] = np.nan
                    test_row[f'{target_col}_rolling_max_{window}'] = np.nan
            # Trend features
            test_row[f'{target_col}_diff_1'] = test_row[f'{target_col}_lag_1'] - test_row.get(f'{target_col}_lag_2', 0)
            test_row[f'{target_col}_diff_7'] = test_row[f'{target_col}_lag_1'] - test_row.get(f'{target_col}_lag_7', 0)
            # Seasonal features
            test_row['day_of_week'] = next_date.dayofweek
            test_row['quarter'] = next_date.quarter
            test_row['is_weekend'] = int(next_date.dayofweek >= 5)
            test_row['week_of_year'] = next_date.isocalendar()[1]
            # Remove target col (to forecast)
            test_row[target_col] = np.nan
            # Prepare final row
            X_pred = pd.DataFrame([test_row[feature_info['features']]])
            y_pred = model_pipeline.predict(X_pred)[0]
            test_row[target_col] = y_pred
            # Append prediction
            future_rows.append({
                store_col: test_row[store_col],
                family_col: test_row[family_col],
                date_col: next_date,
                'forecast': y_pred
            })
            # Add new row to last_df for rolling window
            last_df = pd.concat([last_df, pd.DataFrame([test_row])], ignore_index=True)
    return pd.DataFrame(future_rows)

def evaluate_model_metrics(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return dict(RMSE=rmse, MAE=mae, R2=r2)