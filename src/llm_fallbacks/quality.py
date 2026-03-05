"""Quality scoring heuristic for LLM models.

Computes a transparent capability-based quality score (0-100) from
observable model spec fields.  The score is a weighted sum of feature
flags and log-scaled capacity metrics, normalised to [0, 100].

The heuristic is labelled ``heuristic_v1`` so downstream consumers can
track which version of the formula produced a given score.
"""

from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

#: Maximum raw score before normalisation (sum of all possible points).
_MAX_RAW_SCORE: float = 95.0

#: Version label attached to every score produced by this module.
QUALITY_SOURCE: str = "heuristic_v1"

# Boolean capability bonuses: (spec_key, points)
_CAPABILITY_BONUSES: list[tuple[str, float]] = [
    ("supports_function_calling", 10.0),
    ("supports_vision", 8.0),
    ("supports_response_schema", 7.0),
    ("supports_tool_choice", 5.0),
    ("supports_system_messages", 5.0),
    ("supports_parallel_function_calling", 5.0),
    ("supports_prompt_caching", 3.0),
    ("supports_audio_input", 3.0),
    ("supports_audio_output", 3.0),
    ("supports_pdf_input", 3.0),
    ("supports_assistant_prefill", 3.0),
]


def compute_quality_score(model_spec: dict[str, Any]) -> tuple[float, str]:
    """Compute a quality score (0-100) from observable model capabilities.

    Parameters
    ----------
    model_spec:
        A dictionary conforming to the ``LiteLLMBaseModelSpec`` shape
        (or any dict with the same keys).

    Returns
    -------
    tuple[float, str]
        ``(score, quality_source)`` where *score* is in [0, 100] and
        *quality_source* is a version label for the scoring formula.
    """
    raw: float = 0.0

    # --- Context window (max 30 pts) ---
    max_input = model_spec.get("max_input_tokens") or model_spec.get("max_tokens") or 0
    if isinstance(max_input, (int, float)) and max_input > 0:
        # 4K → 0 pts, 8K → 5 pts, 32K → 15 pts, 128K → 25 pts, 1M → 30 pts cap
        raw += min(30.0, max(0.0, 5.0 * math.log2(max_input / 4096))) if max_input > 4096 else 0.0

    # --- Max output tokens (max 10 pts) ---
    max_output = model_spec.get("max_output_tokens") or 0
    if isinstance(max_output, (int, float)) and max_output > 0:
        raw += min(10.0, max(0.0, 2.5 * math.log2(max_output / 4096))) if max_output > 4096 else 0.0

    # --- Boolean capability bonuses ---
    for key, points in _CAPABILITY_BONUSES:
        if model_spec.get(key):
            raw += points

    # --- Normalise to 0-100 ---
    score = min(100.0, raw * 100.0 / _MAX_RAW_SCORE)
    return round(score, 2), QUALITY_SOURCE
