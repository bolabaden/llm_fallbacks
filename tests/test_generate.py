"""Tests for the config artifact generation pipeline."""

from __future__ import annotations

import pytest

from llm_fallbacks.generate_configs import build_free_models_list, is_local_model

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_FREE_MODELS = [
    (
        "gemini/gemini-2.0-flash",
        {
            "litellm_provider": "gemini",
            "mode": "chat",
            "max_input_tokens": 1_000_000,
            "max_output_tokens": 8192,
            "supports_function_calling": True,
            "supports_vision": True,
            "supports_system_messages": True,
            "supports_response_schema": True,
            "supports_tool_choice": True,
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        },
    ),
    (
        "openai/gpt-4o-mini-free",
        {
            "litellm_provider": "openai",
            "mode": "chat",
            "max_input_tokens": 128_000,
            "max_output_tokens": 4096,
            "supports_function_calling": True,
            "supports_vision": False,
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        },
    ),
    (
        "ollama/llama3",
        {
            "litellm_provider": "ollama",
            "mode": "chat",
            "max_input_tokens": 8192,
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        },
    ),
    (
        "vertex_ai/text-embedding-004",
        {
            "litellm_provider": "vertex_ai",
            "mode": "embedding",
            "max_input_tokens": 2048,
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        },
    ),
]


# ---------------------------------------------------------------------------
# is_local_model tests
# ---------------------------------------------------------------------------


class TestIsLocalModel:
    def test_ollama_is_local(self):
        assert is_local_model("ollama/llama3") is True

    def test_vllm_is_local(self):
        assert is_local_model("vllm/mistral") is True

    def test_lmstudio_is_local(self):
        assert is_local_model("lmstudio/codellama") is True

    def test_xinference_is_local(self):
        assert is_local_model("xinference/chatglm") is True

    def test_openai_is_not_local(self):
        assert is_local_model("openai/gpt-4") is False

    def test_gemini_is_not_local(self):
        assert is_local_model("gemini/gemini-2.0-flash") is False


# ---------------------------------------------------------------------------
# build_free_models_list tests
# ---------------------------------------------------------------------------


class TestBuildFreeModelsList:
    def test_excludes_local_models(self):
        """Local-only models (ollama/) should be excluded."""
        result = build_free_models_list(FIXTURE_FREE_MODELS)
        ids = [entry["id"] for entry in result]
        assert "ollama/llama3" not in ids

    def test_includes_online_models(self):
        """Non-local models should be included."""
        result = build_free_models_list(FIXTURE_FREE_MODELS)
        ids = [entry["id"] for entry in result]
        assert "gemini/gemini-2.0-flash" in ids
        assert "openai/gpt-4o-mini-free" in ids

    def test_sorted_by_quality_score_desc(self):
        """Entries should be sorted by quality_score descending."""
        result = build_free_models_list(FIXTURE_FREE_MODELS)
        scores = [entry["quality_score"] for entry in result]
        assert scores == sorted(scores, reverse=True)

    def test_stable_sort_by_id(self):
        """Entries with the same quality_score should be sorted by id."""
        # Create two models with identical features
        models = [
            ("beta/model-b", {"litellm_provider": "beta", "mode": "chat"}),
            ("alpha/model-a", {"litellm_provider": "alpha", "mode": "chat"}),
        ]
        result = build_free_models_list(models)
        ids = [entry["id"] for entry in result]
        # Both have score 0, so alphabetical by id
        assert ids == ["alpha/model-a", "beta/model-b"]

    def test_schema_fields_present(self):
        """Every entry should have the expected set of keys."""
        result = build_free_models_list(FIXTURE_FREE_MODELS)
        expected_keys = {
            "id",
            "provider",
            "mode",
            "is_free",
            "input_cost_per_token",
            "output_cost_per_token",
            "context_length",
            "max_output_tokens",
            "supports_vision",
            "supports_function_calling",
            "supports_tool_choice",
            "supports_response_schema",
            "supports_system_messages",
            "supports_audio_input",
            "supports_audio_output",
            "supports_pdf_input",
            "supports_prompt_caching",
            "quality_score",
            "quality_source",
        }
        for entry in result:
            assert set(entry.keys()) == expected_keys

    def test_is_free_always_true(self):
        """All entries in the free models list should have is_free=True."""
        result = build_free_models_list(FIXTURE_FREE_MODELS)
        for entry in result:
            assert entry["is_free"] is True

    def test_quality_source_label(self):
        """quality_source should be 'heuristic_v1'."""
        result = build_free_models_list(FIXTURE_FREE_MODELS)
        for entry in result:
            assert entry["quality_source"] == "heuristic_v1"

    def test_empty_input(self):
        """Empty input should produce empty output."""
        result = build_free_models_list([])
        assert result == []

    def test_all_local_models_returns_empty(self):
        """If all models are local, result should be empty."""
        local_only = [
            ("ollama/llama3", {"litellm_provider": "ollama", "mode": "chat"}),
            ("vllm/mistral", {"litellm_provider": "vllm", "mode": "chat"}),
        ]
        result = build_free_models_list(local_only)
        assert result == []
