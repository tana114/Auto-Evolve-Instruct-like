import warnings
import os

from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any

from langchain_core._api.beta_decorator import LangChainBetaWarning

# InMemoryRateLimiter使用時の警告を消す
warnings.filterwarnings("ignore", category=LangChainBetaWarning)

from dotenv import load_dotenv

load_dotenv()

valid_model_names = Literal[
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma-7b-it",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "llama-3.2-1b-preview",
    "llama-3.2-3b-preview",
    "llama-3.2-11b-vision-preview",
    # 'llama-3.2-90b-vision-preview',
]

from langchain_groq import ChatGroq
from langchain_core.rate_limiters import InMemoryRateLimiter

class GroqChatBase(ChatGroq):
    def __init__(
            self,
            model_name: Literal[valid_model_names],
            requests_per_second: Optional[float] = None,
            **kwargs,
    ):
        api_key = os.getenv("GROQ_API_KEY")
        shared_kwargs = dict(
            api_key=api_key,
            model_name=model_name,
            **kwargs,
        )

        if requests_per_second:
            r_limiter = InMemoryRateLimiter(requests_per_second=requests_per_second)
            shared_kwargs["rate_limiter"] = r_limiter

        super().__init__(**shared_kwargs)


if __name__ == "__main__":
    """
    python -m model.groq_llm
    """

    llm = GroqChatBase(
        model_name="llama-3.1-70b-versatile",
        requests_per_second=0.32,
    )

    res = llm.invoke("hello")
    print(res)

    res = llm.invoke("i am tired.")
    print(res)
