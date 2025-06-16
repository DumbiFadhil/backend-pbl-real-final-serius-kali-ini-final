import os
import requests
from dotenv import load_dotenv
import re
from datetime import datetime, timedelta
import joblib

load_dotenv()

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"   # or another model supported by DeepSeek

def strip_think_tags(response: str) -> str:
    """
    Removes <think>...</think> blocks from the LLM response, including any whitespace after.
    """
    return re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL)


# HF_TOKEN = os.environ.get("HF_TOKEN")
# DEEPSEEK_URL = "https://router.huggingface.co/sambanova/v1/chat/completions"
# DEEPSEEK_MODEL = "DeepSeek-R1-0528"

def _parse_query_for_filters(query):
    """
    Naive parser for extracting family and dates from the query.
    Returns a dict: {'family': str or None, 'start_date': str or None, 'end_date': str or None}
    """
    family = None
    start_date = None
    end_date = None

    # Detect 'next week' style
    match_family = re.search(r"\bfor ([A-Z]+)\b", query)
    if match_family:
        family = match_family.group(1)

    if "next week" in query.lower():
        # Assume 'next week' means 7 days after the latest date in data
        return {'family': family, 'period': 'next_week'}
    elif "this week" in query.lower():
        return {'family': family, 'period': 'this_week'}
    else:
        # Optionally, add parsing for explicit dates
        return {'family': family, 'period': None}

def _get_date_range_for_period(forecast_data, period):
    """
    Given a period name, return (start_date, end_date) as datetime objects.
    """
    all_dates = sorted({
        (datetime.strptime(row['date'][:16], "%a, %d %b %Y") if isinstance(row['date'], str) else row['date'])
        for row in forecast_data
    })
    if not all_dates:
        return None, None
    min_date, max_date = all_dates[0], all_dates[-1]
    if period == "next_week":
        start = max_date + timedelta(days=1)
        end = start + timedelta(days=6)
    elif period == "this_week":
        end = max_date
        start = end - timedelta(days=6)
    else:
        return None, None
    # Clamp to available data
    start = max(start, min_date)
    end = min(end, max_date)
    return start, end

def filter_forecast_data(forecast_data, family=None, start_date=None, end_date=None):
    """
    Filter forecast data by family and date range.
    start_date/end_date must be datetime or None.
    """
    filtered = []
    for row in forecast_data:
        # Support both string and Timestamp for row['date']
        row_date = row['date']
        if isinstance(row_date, str):
            try:
                row_date_dt = datetime.strptime(row_date[:16], "%a, %d %b %Y")
            except Exception:
                # fallback: try ISO
                row_date_dt = datetime.fromisoformat(row_date)
        else:
            # Assume Timestamp or datetime
            row_date_dt = row_date

        if family and row['family'].upper() != family.upper():
            continue
        if start_date and end_date:
            if not (start_date <= row_date_dt <= end_date):
                continue
        filtered.append(row)
    return filtered

def summarize_forecast(filtered_data):
    """
    Aggregate total forecast for the filtered data.
    """
    if not filtered_data:
        return None
    total = sum(float(row['forecast']) for row in filtered_data)
    return total

def build_forecast_summary(filtered_data, family, start_date, end_date):
    """
    Build a summary of the forecast for the query/filter.
    """
    total = summarize_forecast(filtered_data)
    if total is None:
        return "No forecast data for the specified criteria."
    if isinstance(start_date, datetime):
        start_date_str = start_date.strftime('%Y-%m-%d')
    else:
        start_date_str = str(start_date)
    if isinstance(end_date, datetime):
        end_date_str = end_date.strftime('%Y-%m-%d')
    else:
        end_date_str = str(end_date)
    summary = (f"**Total predicted sales for {family} from {start_date_str} to {end_date_str}: "
               f"{total:.2f} units (across {len({row['store_nbr'] for row in filtered_data})} stores).**\n")
    # Optionally, add per-store breakdown...
    per_store = {}
    for row in filtered_data:
        store = row['store_nbr']
        per_store.setdefault(store, 0)
        per_store[store] += float(row['forecast'])
    summary += "Per-store breakdown:\n"
    for store, subtotal in sorted(per_store.items()):
        summary += f"- Store {store}: {subtotal:.2f} units\n"
    return summary

def answer_prediction_query(query: str, model_bundle: dict) -> str:
    """
    Filters and aggregates forecast data based on the query,
    then sends the summary and context to DeepSeek for business recommendations/explanation.
    """
    if not DEEPSEEK_API_KEY:
        return "DeepSeek API token (DEEPSEEK_API_KEY) is not set."

    forecast_data = model_bundle.get("last_forecast")
    if not forecast_data:
        return "No forecast data available. Please run a forecast first."

    filters = _parse_query_for_filters(query)
    family = filters['family']
    period = filters['period']

    # Get date range for period
    start_date, end_date = None, None
    if period:
        start_date, end_date = _get_date_range_for_period(forecast_data, period)

    filtered_data = filter_forecast_data(forecast_data, family=family, start_date=start_date, end_date=end_date)

    # Build a summary for the LLM (to ensure it has all relevant info, not a sample)
    summary = build_forecast_summary(filtered_data, family or "ALL", start_date or "start", end_date or "end")

    # Optionally: If you want to give the LLM the data table too, you can build it here:
    def build_table(data):
        lines = ["| date | store_nbr | family | forecast |"]
        lines.append("|------|-----------|--------|----------|")
        for row in data:
            lines.append(
                f"| {row['date']} | {row['store_nbr']} | {row['family']} | {row['forecast']:.2f} |"
            )
        return "\n".join(lines)
    table = build_table(filtered_data[:40])  # If you want to show up to 40 rows for clarity

    system_message = (
        "You are a sales prediction and business recommendation assistant. "
        "ONLY answer questions based on the provided forecast summary and data table. "
        "If the user's question is not about these predictions, politely refuse. "
        "First, review the summary and data, and then suggest actionable business steps (such as restocking, promotions, etc.) if warranted.\n\n"
        "Forecast summary:\n"
        f"{summary}\n\n"
        "Data table (for reference):\n"
        f"{table}"
    )

    payload = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ],
        "model": DEEPSEEK_MODEL,
        "stream": False
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        if "choices" in result and len(result["choices"]) > 0:
            raw_answer = result["choices"][0]["message"]["content"]
            clean_answer = strip_think_tags(raw_answer)
            return clean_answer
        else:
            return "No answer returned from DeepSeek."
    except Exception as e:
        return f"Error contacting DeepSeek: {e}"