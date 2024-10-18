import warnings
import os

from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any

from langchain_openai import AzureChatOpenAI
from langchain_core._api.beta_decorator import LangChainBetaWarning
from langchain_core.rate_limiters import InMemoryRateLimiter

# InMemoryRateLimiter使用時の警告を消す
warnings.filterwarnings("ignore", category=LangChainBetaWarning)

from dotenv import load_dotenv
load_dotenv()

# deployment name
valid_model_names = Literal[
    "**********",  # Specify the deployment name.
]

class AzureChatBase(AzureChatOpenAI):
    def __init__(
        self,
        model_name: Literal[valid_model_names],
        api_version: str,  # e.g. '2024-05-01-preview'
        requests_per_second: Optional[float] = None,
        **kwargs,
    ):
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        shared_kwargs = dict(
            api_key=api_key,
            model_name=model_name,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            **kwargs,
        )

        if requests_per_second:
            r_limiter = InMemoryRateLimiter(requests_per_second=requests_per_second)
            shared_kwargs["rate_limiter"] = r_limiter

        super().__init__(**shared_kwargs)


if __name__ == "__main__":

    """
    python -m model.azure_openai_llm
    """

    llm = AzureChatBase(
        model_name="gpt-4o-mini",
        api_version="2024-05-01-preview",
        # requests_per_second=0.32,
    )

    res = llm.invoke("hello")
    print(res)

    # res = llm.invoke("i am tired.")
    # print(res)
