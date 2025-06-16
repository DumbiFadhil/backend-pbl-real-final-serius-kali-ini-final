import pandas as pd
from transformers import TapasTokenizer, TapasForQuestionAnswering

# Load model and tokenizer once at module level
MODEL_NAME = "google/tapas-base-finetuned-wtq"
tokenizer = TapasTokenizer.from_pretrained(MODEL_NAME)
model = TapasForQuestionAnswering.from_pretrained(MODEL_NAME)

def answer_table_question(query: str, model_bundle: dict) -> str:
    """
    Answers a question about the last forecast table using TAPAS.
    """
    forecast_data = model_bundle.get("last_forecast")
    if not forecast_data:
        return "No forecast data available. Please run a forecast first."

    # Convert to DataFrame, keeping only relevant columns for TAPAS
    df = pd.DataFrame(forecast_data)
    # Optionally, keep only a subset if the table is too large for TAPAS
    if len(df) > 40:
        df = df.head(40)

    # TAPAS can handle only string columns and limited numerics
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str)
    queries = [query]
    inputs = tokenizer(table=df, queries=queries, padding="max_length", return_tensors="pt")
    outputs = model(**inputs)
    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
        inputs, outputs.logits.detach(), outputs.aggregation_logits.detach()
    )
    # Extract answers
    answers = []
    for coordinates in predicted_answer_coordinates:
        if not coordinates:
            answers.append("No answer found.")
        else:
            cell_values = [df.iat[row, column] for row, column in coordinates]
            answers.append(", ".join(map(str, cell_values)))
    return answers[0]