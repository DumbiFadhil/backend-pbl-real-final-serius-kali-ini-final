import requests
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.base import LLM
from pandasai.responses.response_parser import ResponseParser
from pandasai.helpers.logger import Logger

class PhiMiniLLM(LLM):
    def __init__(self, model: str = "phi3:mini"):
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"

    @property
    def type(self):
        return "phi3-mini"

    def call(self, prompt, context: dict = None):
        if not isinstance(prompt, str):
            prompt = str(prompt)

        prompt = (
            "You are a pandas expert. Use only the DataFrame `df` with columns:\n"
            "['store_nbr', 'family', 'forecast_avg', 'historical_avg', 'growth_percent', 'restock_recommendation_percent']\n"
            "DO NOT include any charts, plots, or print statements.\n"
            "Only assign the final result to a variable named `result`.\n"
            "Wrap your code with ```python ... ```.\n\n"
            f"Question:\n{prompt}"
        )

        response = requests.post(
            self.api_url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )

        if response.status_code != 200:
            raise Exception(f"LLM error: {response.text}")

        self.last_response = response.json()["response"]
        return self.last_response


def run_secure_qa_on_forecast(df: pd.DataFrame, question: str):
    llm = PhiMiniLLM()

    config = {
        "llm": llm,
        "enable_cache": False,
        "save_charts": False,
        "enable_memory": False,
        "custom_whitelisted_libraries": ["pandas"],
        "is_conversational_answer": False,
        "logger": Logger()
    }

    sdf = SmartDataframe(df, config=config)
    raw_response = llm.call(question)

    try:
        result = sdf.chat(question)
        if result is None:
            raise ValueError("No result returned")
        if isinstance(result, pd.DataFrame):
            return result, raw_response
        if isinstance(result, list):
            return pd.DataFrame(result), raw_response
        return pd.DataFrame([{"answer": result}]), raw_response
    except Exception as e:
        raise RuntimeError(f"QA execution failed: {str(e)}\nRaw LLM response:\n{raw_response}")
