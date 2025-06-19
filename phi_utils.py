import requests

def phi_straight_answer(question: str, model: str = "phi3:mini"):
    """
    Ask Phi3 via Ollama for a direct answer, but instruct the LLM to ONLY use the available forecast/stock recommendation data.
    Returns the raw LLM output as a string.
    """
    api_url = "http://localhost:11434/api/generate"
    prompt = (
        "You are a helpful business analytics assistant. "
        "You have access ONLY to the provided forecast and stock recommendation data, which contains columns like 'family', 'forecast_avg', 'growth_percent', 'historical_avg', 'restock_recommendation_percent', and 'store_nbr'. "
        "DO NOT use any outside knowledge. "
        "DO NOT make up facts, numbers, supply levels, or any data not present directly in the provided dataset. "
        "DO NOT speculate, estimate, or reference supply, demand, or other business factors unless those values are explicitly present in the data. "
        "If the answer is not directly available from the data, say: 'The data does not contain enough information to answer this question.' "
        "When answering, refer only to the actual column values in the data. "
        "Do NOT include any code in your response.\n\n"
        f"Question: {question}"
    )
    response = requests.post(
        api_url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    if response.status_code != 200:
        return f"LLM error: {response.text}"
    return response.json()["response"]