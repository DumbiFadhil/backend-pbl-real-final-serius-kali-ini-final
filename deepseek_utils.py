import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")  # Huggingface token from .env

DEEPSEEK_URL = "https://router.huggingface.co/sambanova/v1/chat/completions"
DEEPSEEK_MODEL = "DeepSeek-R1-0528"

def answer_prediction_query(query: str, model_bundle: dict) -> str:
    """
    Calls Huggingface's DeepSeek endpoint to answer the query.
    Only allows queries about the prediction.
    """
    if not HF_TOKEN:
        return "Huggingface API token (HF_TOKEN) is not set."

    payload = {
        "messages": [
            {"role": "user", "content": query}
        ],
        "model": DEEPSEEK_MODEL,
        "stream": False
    }
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        # The text is in result["choices"][0]["message"]["content"]
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return "No answer returned from DeepSeek."
    except Exception as e:
        return f"Error contacting DeepSeek: {e}"