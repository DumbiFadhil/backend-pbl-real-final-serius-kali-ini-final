import os
import uuid
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from model_utils import (
    train_xgboost_model,
    forecast_xgboost,
    allowed_file,
    get_model_path,
    evaluate_model_metrics,
    load_xgboost_pipeline
)
from deepseek_utils import answer_prediction_query  # Placeholder for Huggingface/DeepSeek

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

@app.route('/train', methods=['POST'])
def train():
    # Expect: csv file, formdata with params like target_col, date_col, store_col, family_col, etc.
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
    file.save(filepath)

    # Get parameters from form
    target_col = request.form.get('target_col', 'sales')
    date_col = request.form.get('date_col', 'date')
    store_col = request.form.get('store_col', 'store_nbr')
    family_col = request.form.get('family_col', 'family')
    lag_days = int(request.form.get('lag_days', 14))
    test_size = float(request.form.get('test_size', 0.2))

    # Train model
    try:
        (
            model_pipeline, metrics, last_df_lag, feature_info
        ) = train_xgboost_model(
            filepath,
            target_col=target_col,
            date_col=date_col,
            store_col=store_col,
            family_col=family_col,
            lag_days=lag_days,
            test_size=test_size
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Save model
    model_id = str(uuid.uuid4())
    model_path = get_model_path(app.config['MODEL_FOLDER'], model_id)
    joblib.dump({
        'model': model_pipeline,
        'feature_info': feature_info,
        'target_col': target_col,
        'date_col': date_col,
        'store_col': store_col,
        'family_col': family_col,
        'lag_days': lag_days,
        'last_df_lag': last_df_lag
    }, model_path)

    return jsonify({
        'message': 'Model trained and saved',
        'model_id': model_id,
        'metrics': metrics,
        'feature_info': feature_info
    })


@app.route('/forecast', methods=['POST'])
def forecast():
    # Expect: model_id, number_of_weeks, optionally: new data
    data = request.form if request.form else request.json
    model_id = data.get('model_id')
    n_weeks = int(data.get('n_weeks', 2))

    model_path = get_model_path(app.config['MODEL_FOLDER'], model_id)
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not found'}), 404

    model_bundle = joblib.load(model_path)
    model_pipeline = model_bundle['model']
    feature_info = model_bundle['feature_info']
    last_df_lag = model_bundle['last_df_lag']

    # Optionally, take new data to append for rolling forecasting
    # But by default, forecast based on last available lagged data
    try:
        forecast_df = forecast_xgboost(
            model_pipeline, last_df_lag, feature_info,
            n_periods=n_weeks * 7,  # 1 week = 7 days
            target_col=model_bundle['target_col'],
            date_col=model_bundle['date_col'],
            store_col=model_bundle['store_col'],
            family_col=model_bundle['family_col'],
            lag_days=model_bundle['lag_days']
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'forecast': forecast_df.to_dict(orient='records'),
        'message': f'Successfully forecasted for {n_weeks} week(s)'
    })


@app.route('/deepseek', methods=['POST'])
def deepseek():
    # Expect: model_id, query (about prediction)
    data = request.form if request.form else request.json
    model_id = data.get('model_id')
    query = data.get('query', '')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    model_path = get_model_path(app.config['MODEL_FOLDER'], model_id)
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not found'}), 404

    model_bundle = joblib.load(model_path)
    # Only allow queries about the prediction
    nlp_answer = answer_prediction_query(query, model_bundle)
    return jsonify({'answer': nlp_answer})


if __name__ == '__main__':
    app.run(debug=True)