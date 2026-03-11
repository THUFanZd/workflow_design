# agents/runtime.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agents import OpenAIChatCompletionsModel

from .openai_compat import OpenAICompatConfig


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    compat: OpenAICompatConfig


OPENAI_SPEC = ProviderSpec(
    name="openai",
    compat=OpenAICompatConfig(
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
        default_base_url="https://api.openai.com/v1",
    ),
)

DEEPSEEK_SPEC = ProviderSpec(
    name="deepseek",
    compat=OpenAICompatConfig(
        api_key_env="DEEPSEEK_API_KEY",
        base_url_env="DEEPSEEK_BASE_URL",
        default_base_url="https://api.ppio.com/openai",
    ),
)

ZHIPU_SPEC = ProviderSpec(
    name="zhipu",
    compat=OpenAICompatConfig(
        api_key_env="ZHIPU_API_KEY",
        base_url_env="ZHIPU_BASE_URL",
        default_base_url="https://api.ppio.com/openai",
    ),
)


def infer_provider(model_name: str) -> ProviderSpec:
    """
    Keep this logic small and explicit.
    """
    m = model_name.lower()

    if m.startswith(("gpt", "o1", "o3", "o4")) or m.startswith(("pa/gpt", "pa/o")):
        return OPENAI_SPEC

    if m.startswith("deepseek/"):
        return DEEPSEEK_SPEC

    # 你旧代码把 glm / zai-org/glm 归到 zhipu
    if m.startswith(("glm", "zai-org/glm")):
        return ZHIPU_SPEC

    # 兜底：仍当作 OpenAI-compatible（只要你配了 base_url/api_key）
    return OPENAI_SPEC


def build_agents_sdk_model(
    model_name: str,
    *,
    provider: Optional[ProviderSpec] = None,
) -> OpenAIChatCompletionsModel:
    spec = provider or infer_provider(model_name)
    # print(f"Building model for {model_name} with provider {spec.name}")
    client = spec.compat.build_client()
    # print(type(client))
    # print(f"Client built with base_url={client.base_url}")
    return OpenAIChatCompletionsModel(model=model_name, openai_client=client)
