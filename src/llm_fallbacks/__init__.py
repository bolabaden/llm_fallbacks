"""LLM Fallbacks - A library for managing fallbacks for LLM API calls.

This library provides utilities to manage fallback mechanisms for LLM API calls
using the litellm library. It helps you handle API failures gracefully by
providing alternative models to try when a primary model fails.
"""

from llm_fallbacks.core import (
    calculate_cost_per_token,
    get_audio_input_models,
    get_audio_output_models,
    get_audio_speech_models,
    get_audio_transcription_models,
    get_chat_models,
    get_completion_models,
    get_embedding_models,
    get_fallback_list,
    get_function_calling_models,
    get_image_generation_models,
    get_image_input_models,
    get_litellm_model_specs,
    get_litellm_models,
    get_models,
    get_moderation_models,
    get_parallel_function_calling_models,
    get_pdf_input_models,
    get_rerank_models,
    get_vision_models,
    sort_models_by_cost_and_limits,
)
from llm_fallbacks.filter_litellm import filter_models

__version__ = "0.1.0"
__all__ = [
    "calculate_cost_per_token",
    "filter_models",
    "get_audio_input_models",
    "get_audio_output_models",
    "get_audio_speech_models",
    "get_audio_transcription_models",
    "get_chat_models",
    "get_completion_models",
    "get_embedding_models",
    "get_fallback_list",
    "get_function_calling_models",
    "get_image_generation_models",
    "get_image_input_models",
    "get_litellm_model_specs",
    "get_litellm_models",
    "get_models",
    "get_moderation_models",
    "get_parallel_function_calling_models",
    "get_pdf_input_models",
    "get_rerank_models",
    "get_vision_models",
    "sort_models_by_cost_and_limits",
]
