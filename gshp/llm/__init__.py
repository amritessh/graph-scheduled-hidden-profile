"""Local / OpenAI-compatible LLM clients (vLLM, etc.)."""

from gshp.llm.logging_client import LoggingLLMClient
from gshp.llm.openai_local import OpenAICompatibleChat, make_llm_client, parse_model_spec
from gshp.llm.stub_client import StubLLM, parse_choice_json

__all__ = [
    "LoggingLLMClient",
    "OpenAICompatibleChat",
    "StubLLM",
    "make_llm_client",
    "parse_choice_json",
    "parse_model_spec",
]
