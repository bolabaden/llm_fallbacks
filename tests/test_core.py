from llm_fallbacks.config import LiteLLMBaseModelSpec
import pytest
from llm_fallbacks import get_chat_models, get_completion_models, get_embedding_models, get_fallback_list, filter_models


def test_get_chat_models():
    """Test that get_chat_models returns a dictionary."""
    models: dict[str, LiteLLMBaseModelSpec] = get_chat_models()
    assert isinstance(models, dict)


def test_get_completion_models():
    """Test that get_completion_models returns a dictionary."""
    models: dict[str, LiteLLMBaseModelSpec] = get_completion_models()
    assert isinstance(models, dict)


def test_get_embedding_models():
    """Test that get_embedding_models returns a dictionary."""
    models: dict[str, LiteLLMBaseModelSpec] = get_embedding_models()
    assert isinstance(models, dict)


def test_get_fallback_list():
    """Test that get_fallback_list returns a list."""
    fallbacks: list[str] = get_fallback_list("chat")
    assert isinstance(fallbacks, list)


def test_filter_models():
    """Test that filter_models returns a list."""
    models: list[str] = filter_models(model_type="chat")
    assert isinstance(models, list)


def test_filter_models_with_criteria():
    """Test that filter_models with criteria returns a list."""
    models: list[str] = filter_models(model_type="chat", free_only=True)
    assert isinstance(models, list)
