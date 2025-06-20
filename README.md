# Flask XGBoost Time Series API

## Features

- Train an XGBoost time series regressor on sales data via `/train` API
- Forecast future sales via `/forecast` API
- Ask questions about predictions via **TAPAS**-powered `/tapas` API (Tabular QA over forecasts)
- (Legacy) DeepSeek LLM question answering via `/deepseek` API (now optional; replaced by TAPAS for table reasoning)
- Get automatic **stock recommendations** via `/stock_recommendation` API (restocking logic)
- **LLM-powered QA** via `/qa` API using Ollama and phi3:mini (natural language analytics and summaries)
- Modular: Can be used with any time series sales data, just specify the column names

---

## Requirements

- Python 3.8+
- `torch`, `transformers`, `pandas`, `flask`, `joblib`, `pandasai`
- For TAPAS: PyTorch and Huggingface Transformers
- **For /qa endpoint:** [Ollama](https://ollama.com/) running `phi3:mini` locally (see below)

Install dependencies:
```bash
pip install -r requirements.txt
```

### Ollama + phi3:mini

You must have [Ollama](https://ollama.com/download) running locally and have pulled the phi3:mini model:
```bash
ollama run phi3:mini
```
This is required for the `/qa` endpoint to work.

---

## Running the API

```bash
python app.py
```
The server should run at `http://localhost:5000` by default.

---

## API Endpoints

### 1. `/train` (POST)

Uploads a CSV file and trains a model.  
**Form-data parameters:**

| Key         | Type   | Example       | Description                           |
|-------------|--------|---------------|---------------------------------------|
| file        | File   | `train.csv`   | Your sales CSV file                   |
| target_col  | String | `sales`       | Target column name                    |
| date_col    | String | `date`        | Date column name                      |
| store_col   | String | `store_nbr`   | Store column name                     |
| family_col  | String | `family`      | Family/category column name           |
| lag_days    | Int    | `14`          | Number of lag days to use             |
| test_size   | Float  | `0.2`         | Test split proportion (0.2 = 20%)     |

**Mock Response:**
```json
{
  "message": "Model trained and saved",
  "model_id": "b82d0c5d-0f25-4b51-bb7b-09e0b3b53b21",
  "metrics": {
    "RMSE": 298.1,
    "Correlation": 0.98,
    "MAE": 46.0,
    "RAE": 0.04,
    "RRSE": 0.15,
    "MAPE": 15.88,
    "R2": 0.97
  },
  "feature_info": {
    "features": ["store_nbr", "family", "date", "lag_1", "lag_2", "..."]
  }
}
```

---

### 2. `/forecast` (POST)

Uses a trained model to forecast future sales.

| Key        | Type   | Example        | Description                               |
|------------|--------|----------------|-------------------------------------------|
| model_id   | String | (from /train)  | The model id returned by `/train`         |
| n_weeks    | Int    | `2`            | Number of weeks to forecast (7 days/week) |

**Mock Response:**
```json
{
  "forecast": [
    {
      "store_nbr": 1,
      "family": "AUTOMOTIVE",
      "date": "2025-06-16",
      "forecast": 123.45
    },
    {
      "store_nbr": 2,
      "family": "GROCERY",
      "date": "2025-06-16",
      "forecast": 234.56
    }
    // ...more rows...
  ],
  "message": "Successfully forecasted for 2 week(s)"
}
```

---

### 3. `/tapas` (POST)

Ask a question about the latest forecast table using TAPAS (tabular QA model).  
Supports questions like "Which product had the highest forecast?", "Which family had the highest growth?", etc.

| Key        | Type   | Example                              | Description                                    |
|------------|--------|--------------------------------------|------------------------------------------------|
| model_id   | String | (from /train)                        | The model id returned by `/train`              |
| question   | String | "Which product had the highest growth?" | The question about predictions/forecast     |

**Mock Response:**
```json
{
  "question": "Which family had the highest forecast?",
  "answer": "GROCERY"
}
```

---

### 4. `/stock_recommendation` (POST)

Returns restocking suggestions based on the latest forecast data, comparing historical and forecasted sales.

| Key        | Type   | Example                 | Description                                    |
|------------|--------|-------------------------|------------------------------------------------|
| model_id   | String | (from /train)           | The model id returned by `/train` (or omit for latest) |

**Request Example:**
```json
{
  "model_id": "4d024e1e-9f9d-4dae-ace8-13db029cd898"
}
```
or just
```json
{}
```
to use the latest model.

**Mock Response:**
```json
{
  "recommendations": [
    {
      "family": "AUTOMOTIVE",
      "forecast_avg": 5.03,
      "growth_percent": 25.69,
      "historical_avg": 4.0,
      "restock_recommendation_percent": 26,
      "store_nbr": 1
    },
    {
      "family": "BEAUTY",
      "forecast_avg": 5.03,
      "growth_percent": 25.69,
      "historical_avg": 4.0,
      "restock_recommendation_percent": 26,
      "store_nbr": 1
    }
    // ... more recommendations ...
  ],
  "total_recommendations": 260
}
```

---

### 5. `/qa` (POST) — NEW!

Ask any analytics question about the forecast or stock recommendation data using an LLM (phi3:mini via Ollama).

| Key        | Type   | Example                            | Description                                    |
|------------|--------|------------------------------------|------------------------------------------------|
| model_id   | String | (from /train) (optional)           | The model id returned by `/train`              |
| question   | String | "Which items should be restocked the most?" | The analytics question to answer        |
| format     | String | "summary" or "records" (optional)  | Response format: "summary" (text) or default "records" (full data) |

**Request Example:**
```json
{
  "question": "Which items should be restocked the most?",
  "format": "summary"
}
```

**Mock Response:**
```json
{
  "data": [
    {
      "family": "AUTOMOTIVE",
      "forecast_avg": 5.03,
      "growth_percent": 151.38,
      "historical_avg": 2.0,
      "restock_recommendation_percent": 151,
      "store_nbr": 2
    },
    {
      "family": "PLAYERS AND ELECTRONICS",
      "forecast_avg": 4.92,
      "growth_percent": 145.82,
      "historical_avg": 2.0,
      "restock_recommendation_percent": 146,
      "store_nbr": 54
    }
  ],
  "llm_raw_response": "```python\n# Assuming 'df' is already defined and contains relevant data including growth percentages.\nmax_growth_items = df[df['growth_percent'] == df['growth_percent'].max()]['family'].tolist()\nresult = max_growth_items  # Assign the final result to 'result' variable without printing or plotting.\n```",
  "question": "Which items should be restocked the most?",
  "summary": "You should restock AUTOMOTIVE (Store 2), HARDWARE (Store 2), HARDWARE (Store 3) due to high growth."
}
```

---

### 6. `/deepseek` (POST) (legacy/optional)

Ask a question about the model's predictions using the legacy DeepSeek LLM (now replaced by TAPAS for table QA, but still available if enabled).

| Key        | Type   | Example                   | Description                                    |
|------------|--------|---------------------------|------------------------------------------------|
| model_id   | String | (from /train)             | The model id returned by `/train`              |
| query      | String | "What is the forecast?"   | The question about predictions/forecast        |

---

### 7. `/latest_model` (GET)

Returns the latest trained model's ID.

---

## Ollama and phi3:mini LLM Setup

To use the `/qa` endpoint, you **must be running Ollama with the phi3:mini model**.  
Install [Ollama](https://ollama.com/download) and pull/run the model:

```bash
ollama run phi3:mini
```
You can then run the Flask API and use `/qa` for advanced analytics/summaries.

---

## Customizing Table QA with TAPAS

- The `/tapas` route uses Huggingface TAPAS for direct table question answering.
- See `tapas_utils.py` for implementation details.
- All tabular data is converted to string for TAPAS compatibility.

## Troubleshooting

- If you see `"error": "'DataFrame' object has no attribute 'append'"` update any `.append()` calls to use `pd.concat()`.
- Ensure your CSV columns match what you pass in the form fields.
- For TAPAS, all table cells are stringified for compatibility.

---

## License

MIT