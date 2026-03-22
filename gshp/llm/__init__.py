"""Local / OpenAI-compatible LLM clients (vLLM, etc.)."""

from gshp.llm.openai_local import OpenAICompatibleChat, parse_model_spec

__all__ = ["OpenAICompatibleChat", "parse_model_spec"]
