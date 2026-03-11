from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI


@dataclass(frozen=True)
class OpenAICompatConfig:
    """
    OpenAI-compatible ChatCompletions provider config.

    Examples:
      - OpenAI:   api_key_env=OPENAI_API_KEY,   base_url_env=OPENAI_BASE_URL
      - DeepSeek: api_key_env=DEEPSEEK_API_KEY, base_url_env=DEEPSEEK_BASE_URL
      - Zhipu:    api_key_env=ZHIPU_API_KEY,    base_url_env=ZHIPU_BASE_URL
    """
    api_key_env: str
    base_url_env: str
    default_base_url: Optional[str] = "https://api.ppio.com/openai"

    def build_client(self) -> AsyncOpenAI:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing env var: {self.api_key_env}")

        base_url = os.getenv(self.base_url_env) or self.default_base_url
        # AsyncOpenAI accepts base_url=None for official OpenAI
        return AsyncOpenAI(api_key=api_key, base_url=base_url)
