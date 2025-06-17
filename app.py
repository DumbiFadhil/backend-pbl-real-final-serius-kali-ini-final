import os
import uuid
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from llm_utils import run_secure_qa_on_forecast

from model_utils import (
    train_xgboost_model,
    forecast_xgboost,
    allowed_file,
    get_model_path,
    evaluate_model_metrics,
    load_xgboost_pipeline
)
from deepseek_utils import answer_prediction_query  # Placeholder for Huggingface/DeepSeek
from utils import load_latest_model_id, save_latest_model_id
from tapas_utils import answer_table_question  # Placeholder for TAPAS

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

    save_latest_model_id(model_id)

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

        # If no model_id provided, load latest
    if not model_id:
        model_id = load_latest_model_id()
        if not model_id:
            return jsonify({'error': 'No model_id provided and no latest model found'}), 400

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
        model_bundle["last_forecast"] = forecast_df.to_dict(orient='records')
        joblib.dump(model_bundle, model_path)
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
    
    # If no model_id provided, load latest
    if not model_id:
        model_id = load_latest_model_id()
        if not model_id:
            return jsonify({'error': 'No model_id provided and no latest model found'}), 400

    model_path = get_model_path(app.config['MODEL_FOLDER'], model_id)
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not found'}), 404

    model_bundle = joblib.load(model_path)
    # Only allow queries about the prediction
    nlp_answer = answer_prediction_query(query, model_bundle)
    return jsonify({'answer': nlp_answer})

@app.route("/tapas", methods=["POST"])
def tapas():
    data = request.form if request.form else request.json
    model_id = data.get('model_id')

    # If no model_id provided, load latest
    if not model_id:
        model_id = load_latest_model_id()
        if not model_id:
            return jsonify({'error': 'No model_id provided and no latest model found'}), 400

    model_path = get_model_path(app.config['MODEL_FOLDER'], model_id)
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not found'}), 404

    # Load model bundle and forecast data
    model_bundle = joblib.load(model_path)
    last_forecast = model_bundle.get('last_forecast')
    if not last_forecast:
        return jsonify({'error': 'No forecast data found. Please run /forecast first.'}), 400

    question = data.get('question')
    if not question:
        return jsonify({'error': 'Missing question in request'}), 400

    try:
        forecast_df = pd.DataFrame(last_forecast)

        # Summarize by store-family-week
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])
        forecast_df["week"] = forecast_df["date"].dt.strftime('%Y-W%U')

        forecast_df = forecast_df.astype(str)
        agg_df = (
            forecast_df.groupby(["store_nbr", "family", "week"])
            .agg(total_forecast=("forecast", "sum"))
            .reset_index()
            .astype(str)
        )

        # Limit to 50 rows
        agg_df = agg_df.head(50)
        answer = answer_table_question(agg_df, question)

        answer = answer_table_question(forecast_df, question)
        return jsonify({
            "question": question,
            "answer": answer
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/stock_recommendation", methods=["POST"])
def stock_recommendation():
    data = request.form if request.form else request.json
    model_id = data.get("model_id")

    if not model_id:
        model_id = load_latest_model_id()
        if not model_id:
            return jsonify({"error": "No model_id provided and no latest model found"}), 400

    model_path = get_model_path(app.config["MODEL_FOLDER"], model_id)
    if not os.path.exists(model_path):
        return jsonify({"error": "Model not found"}), 404

    model_bundle = joblib.load(model_path)
    forecast = model_bundle.get("last_forecast")

    if not forecast:
        return jsonify({"error": "No forecast data found. Please run /forecast first."}), 400

    # Load past real sales data (assumed stored during training)
    last_df_lag = model_bundle.get("last_df_lag")
    if last_df_lag is None or 'sales' not in last_df_lag.columns:
        return jsonify({"error": "No historical sales data available"}), 400

    try:
        forecast_df = pd.DataFrame(forecast)
        past_df = last_df_lag.copy()
        past_df["date"] = pd.to_datetime(past_df["date"])
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])

        recommendations = []

        for (store, family), group in forecast_df.groupby(["store_nbr", "family"]):
            recent_forecast = group.sort_values("date").tail(7)["forecast"].mean()

            past_sales = past_df[
                (past_df["store_nbr"] == store) & (past_df["family"] == family)
            ].sort_values("date").tail(14)

            historical_avg = past_sales["sales"].mean()
            if historical_avg == 0 or np.isnan(historical_avg):
                continue

            growth = (recent_forecast - historical_avg) / historical_avg
            if growth > 0.25:  # 25% growth threshold
                recommendations.append({
                    "store_nbr": int(store),
                    "family": str(family),
                    "forecast_avg": float(round(recent_forecast, 2)),
                    "historical_avg": float(round(historical_avg, 2)),
                    "growth_percent": float(round(growth * 100, 2)),
                    "restock_recommendation_percent": int(round(growth * 100)),
                })

        model_bundle["restock_df"] = recommendations
        joblib.dump(model_bundle, model_path)

        return jsonify({
            "total_recommendations": len(recommendations),
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/qa", methods=["POST"])
def qa():
    data = request.form if request.form else request.json
    model_id = data.get("model_id")
    question = data.get("question")
    response_format = data.get("format", "records").lower()  # "summary" or "records"

    if not question:
        return jsonify({"error": "Missing question"}), 400

    if not model_id:
        model_id = load_latest_model_id()
        if not model_id:
            return jsonify({"error": "No model_id provided and no latest model found"}), 400

    model_path = get_model_path(app.config["MODEL_FOLDER"], model_id)
    if not os.path.exists(model_path):
        return jsonify({"error": "Model not found"}), 404

    model_bundle = joblib.load(model_path)

    # Decide which dataset to query
    if "restock" in question.lower() or "growth" in question.lower():
        raw_data = model_bundle.get("restock_df")
        if not raw_data:
            return jsonify({"error": "No restock data found. Please run /stock_recommendation first."}), 400
        df_to_query = pd.DataFrame(raw_data)
    else:
        raw_data = model_bundle.get("last_forecast")
        if not raw_data:
            return jsonify({"error": "No forecast data found. Please run /forecast first."}), 400
        df_to_query = pd.DataFrame(raw_data)

    if df_to_query.empty:
        return jsonify({
            "question": question,
            "answer": "Sorry, the dataset is empty. Please check your forecast or stock data.",
            "llm_raw_response": None
        })

    try:
        answer, raw_llm = run_secure_qa_on_forecast(df_to_query, question)

        # 🛡️ Defensive handling if LLM fails or returns string
        if not isinstance(answer, pd.DataFrame):
            # Attempt to convert from list of dicts or fallback
            if isinstance(answer, list):
                answer = pd.DataFrame(answer)
            else:
                return jsonify({
                    "question": question,
                    "answer": "Sorry, the model could not generate a valid result.",
                    "llm_raw_response": raw_llm
                })

        # ✅ If user wants summary
        if response_format == "summary":
            if answer.empty:
                return jsonify({
                    "question": question,
                    "summary": "No items meet the restock criteria at this time.",
                    "llm_raw_response": raw_llm,
                    "data": []
                })

            # ✅ Safely build the summary
            summary_parts = []
            for _, row in answer.head(3).iterrows():
                store = row.get("store_nbr", "unknown store")
                family = row.get("family", "unknown item")
                summary_parts.append(f"{family} (Store {store})")

            summary_text = "You should restock " + ", ".join(summary_parts) + " due to high growth."

            return jsonify({
                "question": question,
                "summary": summary_text,
                "llm_raw_response": raw_llm,
                "data": answer.to_dict(orient="records")
            })

        # ✅ Default: full JSON result
        return jsonify({
            "question": question,
            "answer": answer.to_dict(orient="records"),
            "llm_raw_response": raw_llm
        })

    except Exception as e:
        return jsonify({"error": f"Secure QA failed: {str(e)}"}), 500

@app.route('/latest_model', methods=['GET'])
def latest_model():
    model_id = load_latest_model_id()
    if not model_id:
        return jsonify({'error': 'No latest model found'}), 404
    return jsonify({'latest_model_id': model_id})

if __name__ == '__main__':
    app.run(debug=True)