# Base file for Ollama helper
# Holds the strings needed for requests

import os

from dotenv import load_dotenv

try:
    from langchain_community.chat_models import ChatOllama  # type: ignore
except Exception:
    try:
        from langchain_ollama import ChatOllama  # type: ignore
    except Exception:
        ChatOllama = None

try:
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception:
    ChatOpenAI = None

class OllamaHelperBase():
    
    def __init__(self, base_url="http://localhost:11434", model='llama3', temp=0.7, logging=False) -> None:
        self.logging = logging
        if 'gpt' in model:
            if ChatOpenAI is None:
                raise ModuleNotFoundError(
                    "langchain_openai is required for GPT models. Install langchain-openai."
                )
            load_dotenv()
            OPENAI_KEY = os.getenv('OPENAI_KEY')
            self.model = ChatOpenAI(model=model, api_key=OPENAI_KEY, temperature=temp)
        else:
            if ChatOllama is None:
                raise ModuleNotFoundError(
                    "ChatOllama backend not found. Install langchain-community or langchain-ollama."
                )
            self.model = ChatOllama(base_url=base_url, model=model, format="json", temperature=temp)
    
    # Reads the contents of the given file
    def read_python_file(self, file):
        with open(file, 'r') as file:
            data = file.read().replace('\n', '')
        return data
