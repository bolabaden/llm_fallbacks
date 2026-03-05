"""Tests for the quality scoring heuristic."""

from __future__ import annotations

import pytest

from llm_fallbacks.quality import QUALITY_SOURCE, compute_quality_score


class TestComputeQualityScore:
    """Tests for compute_quality_score."""

    def test_empty_spec_returns_zero(self):
        """An empty spec should yield a score of 0."""
        score, source = compute_quality_score({})
        assert score == 0.0
        assert source == QUALITY_SOURCE

    def test_source_is_heuristic_v1(self):
        """Quality source label is always 'heuristic_v1'."""
        _, source = compute_quality_score({"supports_vision": True})
        assert source == "heuristic_v1"

    def test_score_range_zero_to_hundred(self):
        """Score must always be in [0, 100]."""
        # Minimal spec
        score_min, _ = compute_quality_score({})
        assert 0.0 <= score_min <= 100.0

        # Maximal spec (all features + huge context)
        maximal_spec = {
            "max_input_tokens": 2_000_000,
            "max_output_tokens": 200_000,
            "supports_function_calling": True,
            "supports_vision": True,
            "supports_response_schema": True,
            "supports_tool_choice": True,
            "supports_system_messages": True,
            "supports_parallel_function_calling": True,
            "supports_prompt_caching": True,
            "supports_audio_input": True,
            "supports_audio_output": True,
            "supports_pdf_input": True,
            "supports_assistant_prefill": True,
        }
        score_max, _ = compute_quality_score(maximal_spec)
        assert 0.0 <= score_max <= 100.0

    def test_context_window_scaling(self):
        """Larger context windows should produce higher scores."""
        score_4k, _ = compute_quality_score({"max_input_tokens": 4096})
        score_32k, _ = compute_quality_score({"max_input_tokens": 32768})
        score_128k, _ = compute_quality_score({"max_input_tokens": 131072})
        assert score_4k < score_32k < score_128k

    def test_context_below_4k_gets_zero_context_points(self):
        """Models with ≤4K context get 0 context points (but can still score on other features)."""
        score_2k, _ = compute_quality_score({"max_input_tokens": 2048})
        score_4k, _ = compute_quality_score({"max_input_tokens": 4096})
        assert score_2k == score_4k == 0.0

    def test_function_calling_adds_points(self):
        """Function calling support should increase the score."""
        score_without, _ = compute_quality_score({})
        score_with, _ = compute_quality_score({"supports_function_calling": True})
        assert score_with > score_without

    def test_vision_adds_points(self):
        """Vision support should increase the score."""
        score_without, _ = compute_quality_score({})
        score_with, _ = compute_quality_score({"supports_vision": True})
        assert score_with > score_without

    def test_feature_rich_model_scores_high(self):
        """A model with many features should score significantly above 0."""
        spec = {
            "max_input_tokens": 128_000,
            "max_output_tokens": 16_000,
            "supports_function_calling": True,
            "supports_vision": True,
            "supports_response_schema": True,
            "supports_tool_choice": True,
            "supports_system_messages": True,
        }
        score, _ = compute_quality_score(spec)
        assert score > 50.0

    def test_deterministic(self):
        """Same input should always produce the same score."""
        spec = {"max_input_tokens": 64_000, "supports_vision": True, "supports_function_calling": True}
        score1, source1 = compute_quality_score(spec)
        score2, source2 = compute_quality_score(spec)
        assert score1 == score2
        assert source1 == source2

    def test_max_tokens_fallback(self):
        """When max_input_tokens is absent, max_tokens should be used."""
        score_input, _ = compute_quality_score({"max_input_tokens": 32768})
        score_fallback, _ = compute_quality_score({"max_tokens": 32768})
        assert score_input == score_fallback

    def test_output_tokens_add_points(self):
        """Large max_output_tokens should add points."""
        score_without, _ = compute_quality_score({})
        score_with, _ = compute_quality_score({"max_output_tokens": 32768})
        assert score_with > score_without
