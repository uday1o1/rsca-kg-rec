import requests
from dotenv import load_dotenv

load_dotenv()

def query_ollama(prompt: str, model: str = "llama3.2") -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]


if __name__ == "__main__":
    # quick sanity test
    result = query_ollama("Say hello in one sentence.")
    print(result)