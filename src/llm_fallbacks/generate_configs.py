"""Generate machine-consumable configuration artifacts for LLM Fallbacks.

Usage::

    python -m llm_fallbacks.generate_configs --output-dir configs

Artifacts produced (all written to *output-dir*):

* ``all_models.json``          - full model-id to spec map
* ``free_models.json``         - rich array sorted by quality_score (desc)
* ``free_models_ids.txt``      - one model ID per line, same order
* ``custom_providers.json``    - serialised provider configs
* ``litellm_config.yaml``      - LiteLLM proxy config (all models)
* ``litellm_config_free.yaml`` - LiteLLM proxy config (free models only)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import uuid

from pathlib import Path
from typing import Any

if not importlib.util.find_spec("llm_fallbacks"):
    import sys

    sys.path.append(str(Path(__file__).parents[1]))

from llm_fallbacks.config import ALL_MODELS, CUSTOM_PROVIDERS, FREE_MODELS, CustomProviderConfig, LiteLLMYAMLConfig
from llm_fallbacks.core import calculate_cost_per_token, get_model_priority_rank
from llm_fallbacks.quality import compute_quality_score

logger = logging.getLogger(__name__)

# Providers whose models are local-only (not usable via public API)
LOCAL_PROVIDERS = frozenset(["ollama", "vllm", "lmstudio", "xinference"])


def is_local_model(model_id: str) -> bool:
    """Return True if *model_id* belongs to a local-only provider."""
    return any(model_id.startswith(f"{p}/") for p in LOCAL_PROVIDERS)


def build_free_models_list(free_models: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Build the rich ``free_models.json`` payload.

    Parameters
    ----------
    free_models:
        Sorted list of ``(model_id, spec)`` tuples (e.g. from
        ``config.FREE_MODELS``).

    Returns
    -------
    list[dict]
        Array of enriched model objects, sorted by ``quality_score``
        descending then by ``id`` for determinism.
    """
    entries: list[dict[str, Any]] = []
    for model_id, spec in free_models:
        if is_local_model(model_id):
            continue

        quality_score, quality_source = compute_quality_score(spec)
        entry: dict[str, Any] = {
            "id": model_id,
            "provider": spec.get("litellm_provider", ""),
            "mode": spec.get("mode", ""),
            "is_free": True,
            "input_cost_per_token": spec.get("input_cost_per_token", 0),
            "output_cost_per_token": spec.get("output_cost_per_token", 0),
            "context_length": spec.get("max_input_tokens") or spec.get("max_tokens") or 0,
            "max_output_tokens": spec.get("max_output_tokens", 0),
            "supports_vision": bool(spec.get("supports_vision", False)),
            "supports_function_calling": bool(spec.get("supports_function_calling", False)),
            "supports_tool_choice": bool(spec.get("supports_tool_choice", False)),
            "supports_response_schema": bool(spec.get("supports_response_schema", False)),
            "supports_system_messages": bool(spec.get("supports_system_messages", False)),
            "supports_audio_input": bool(spec.get("supports_audio_input", False)),
            "supports_audio_output": bool(spec.get("supports_audio_output", False)),
            "supports_pdf_input": bool(spec.get("supports_pdf_input", False)),
            "supports_prompt_caching": bool(spec.get("supports_prompt_caching", False)),
            "quality_score": quality_score,
            "quality_source": quality_source,
        }
        entries.append(entry)

    # Keep pinned fallback entries ahead of raw quality ranking, then sort by quality descending.
    entries.sort(key=lambda e: (get_model_priority_rank(e["id"]), -e["quality_score"], e["id"]))
    return entries


# ---------------------------------------------------------------------------
# LiteLLM YAML config generation (kept for backwards compatibility)
# ---------------------------------------------------------------------------


def to_litellm_config_yaml(
    providers: list[CustomProviderConfig], free_only: bool = False, online_only: bool = False
) -> LiteLLMYAMLConfig:
    """Convert the provider config to a LiteLLM YAML config format."""
    config: LiteLLMYAMLConfig = {
        "cache": {
            "host": "localhost",
            "mode": "default_off",
            "namespace": "litellm.caching.caching",
            "port": 6379,
            "supported_call_types": ["acompletion", "atext_completion", "aembedding", "atranscription"],
            "ttl": 600,
            "type": "redis",
        },
        "general_settings": {
            "master_key": f"sk-{uuid.uuid4().hex}",
            "alerting": ["slack", "email"],
            "proxy_batch_write_at": 60,
            "database_connection_pool_limit": 10,
            "alerting_threshold": 0,
            "allow_requests_on_db_unavailable": True,
            "allowed_routes": [],
            "background_health_checks": True,
            "database_url": f"postgresql://postgres:{os.environ.get('POSTGRES_PASSWORD')}@localhost:5432/postgres",
            "disable_adding_master_key_hash_to_db": False,
            "disable_master_key_return": False,
            "disable_reset_budget": False,
            "disable_retry_on_max_parallel_request_limit_error": False,
            "disable_spend_logs": False,
            "enable_jwt_auth": False,
            "enforce_user_param": False,
            "global_max_parallel_requests": 0,
            "health_check_interval": 300,
            "infer_model_from_keys": True,
            "max_parallel_requests": 0,
            "use_client_credentials_pass_through_routes": False,
        },
        "litellm_settings": {
            "callbacks": ["otel"],
            "content_policy_fallbacks": [],
            "context_window_fallbacks": [],
            "default_fallbacks": [],
            "failure_callback": ["sentry"],
            "force_ipv4": True,
            "json_logs": True,
            "redact_user_api_key_info": True,
            "request_timeout": 600,
            "service_callbacks": ["datadog", "prometheus"],
            "set_verbose": False,
            "turn_off_message_logging": False,
        },
        "model_list": [],
        "router_settings": {
            "allowed_fails": 3,
            "allowed_fails_policy": {
                "BadRequestErrorAllowedFails": 1000,
                "AuthenticationErrorAllowedFails": 10,
                "TimeoutErrorAllowedFails": 12,
                "RateLimitErrorAllowedFails": 10000,
                "ContentPolicyViolationErrorAllowedFails": 15,
                "InternalServerErrorAllowedFails": 20,
            },
            "cooldown_time": 30,
            "disable_cooldowns": False,
            "enable_pre_call_checks": True,
            "enable_tag_filtering": True,
            "fallbacks": [],
            "retry_policy": {
                "AuthenticationErrorRetries": 3,
                "TimeoutErrorRetries": 3,
                "RateLimitErrorRetries": 3,
                "ContentPolicyViolationErrorRetries": 4,
                "InternalServerErrorRetries": 4,
            },
            "routing_strategy": "simple-shuffle",
        },
    }

    for p in providers:
        for model_name, model_spec in (p.free_models if free_only else p.model_specs).items():
            is_free = calculate_cost_per_token(model_spec) <= 0.0
            is_local = (
                model_name.casefold().startswith("ollama/")
                or model_name.casefold().startswith("vllm/")
                or model_name.casefold().startswith("xinference/")
                or model_name.casefold().startswith("lmstudio/")
                or "127.0.0.1" in p.base_url
                or "localhost" in p.base_url
                or "0.0.0.0" in p.base_url
            )
            if free_only and (not is_free or is_local):
                continue

            key_name = model_name if "/" in model_name else f"{p.provider_name}/{model_name}"
            model_entry = {
                "model_name": key_name,
                "litellm_params": {
                    "model": (key_name if key_name.startswith("openai/") else f"openai/{key_name}"),
                    "api_base": p.base_url,
                    **{"api_key": f"os.environ/{p.api_env_key_name}"},
                    **({} if p.api_version is None else {"api_version": p.api_version}),
                    **dict(model_spec.items()),
                },
            }
            config["model_list"].append(model_entry)  # pyright: ignore[reportArgumentType]

            # Determine suitable fallbacks based on mode and cost
            suitable_fallbacks: list[str] = []
            total_fallbacks_found = 0
            total_fallbacks_required = 25
            for k, v in FREE_MODELS:
                k_name = (
                    k
                    if k.startswith(f"{v.get('litellm_provider')}/") and k not in suitable_fallbacks
                    else f"{v.get('litellm_provider')}/{k}"
                )
                if k.casefold() == model_name.casefold():
                    continue
                if (
                    v.get("mode") is not None
                    and model_spec.get("mode") is not None
                    and v.get("mode") != model_spec.get("mode")
                ):
                    continue
                if (
                    v.get("supports_vision") is not None
                    and model_spec.get("supports_vision") is not None
                    and v.get("supports_vision") != model_spec.get("supports_vision")
                ):
                    continue
                if (
                    v.get("supports_embedding_image_input") is not None
                    and model_spec.get("supports_embedding_image_input") is not None
                    and v.get("supports_embedding_image_input") != model_spec.get("supports_embedding_image_input")
                ):
                    continue
                if (
                    v.get("supports_audio_input") is not None
                    and model_spec.get("supports_audio_input") is not None
                    and v.get("supports_audio_input") != model_spec.get("supports_audio_input")
                ):
                    continue
                if (
                    v.get("supports_audio_output") is not None
                    and model_spec.get("supports_audio_output") is not None
                    and v.get("supports_audio_output") != model_spec.get("supports_audio_output")
                ):
                    continue
                if model_spec.get("mode") is not None and model_spec.get("mode") != "chat":
                    total_fallbacks_required = 125
                if online_only and (
                    k.casefold().startswith("ollama/")
                    or k.casefold().startswith("vllm/")
                    or k.casefold().startswith("xinference/")
                    or k.casefold().startswith("lmstudio/")
                    or "127.0.0.1" in p.base_url
                    or "localhost" in p.base_url
                    or "0.0.0.0" in p.base_url
                ):
                    continue
                suitable_fallbacks.append(k_name)
                total_fallbacks_found += 1
                if total_fallbacks_found >= total_fallbacks_required:
                    break

            if suitable_fallbacks:
                fallback_list = config["router_settings"].setdefault("fallbacks", [])
                fallback_entry = {model_name: suitable_fallbacks}
                fallback_list.append(fallback_entry)

    return config


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------


def generate(output_dir: str = "configs") -> None:
    """Generate all config artifacts into *output_dir*."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. custom_providers.json
    custom_providers_path = out / "custom_providers.json"
    logger.info("Writing %s", custom_providers_path)
    custom_providers_path.write_text(
        json.dumps([provider.to_dict() for provider in CUSTOM_PROVIDERS], indent=4, ensure_ascii=True)
    )

    # 2. all_models.json
    all_models_path = out / "all_models.json"
    logger.info("Writing %s", all_models_path)
    all_models_path.write_text(json.dumps(dict(ALL_MODELS), indent=4, ensure_ascii=True))

    # 3. free_models.json  (rich schema with quality scores)
    free_models_list = build_free_models_list(FREE_MODELS)
    free_models_path = out / "free_models.json"
    logger.info("Writing %s (%d models)", free_models_path, len(free_models_list))
    free_models_path.write_text(json.dumps(free_models_list, indent=2, ensure_ascii=True))

    # 4. free_models_ids.txt
    free_ids_path = out / "free_models_ids.txt"
    logger.info("Writing %s", free_ids_path)
    free_ids_path.write_text("\n".join(entry["id"] for entry in free_models_list) + "\n")

    # 5. LiteLLM YAML configs (optional - requires pyyaml)
    try:
        import yaml

        litellm_config_free_path = out / "litellm_config_free.yaml"
        logger.info("Writing %s", litellm_config_free_path)
        litellm_config_free_path.write_text(
            yaml.dump(to_litellm_config_yaml(CUSTOM_PROVIDERS, free_only=True), sort_keys=False, allow_unicode=True),
            errors="replace",
            encoding="utf-8",
        )

        litellm_config_path = out / "litellm_config.yaml"
        logger.info("Writing %s", litellm_config_path)
        litellm_config_path.write_text(
            yaml.dump(to_litellm_config_yaml(CUSTOM_PROVIDERS, free_only=False), sort_keys=False, allow_unicode=True),
            errors="replace",
            encoding="utf-8",
        )
    except ImportError as e:
        logger.warning("Failed to generate YAML configs (pyyaml not installed): %s: %s", e.__class__.__name__, e)

    logger.info("Done - artifacts written to %s/", out)


def main() -> None:
    """CLI entry-point for ``python -m llm_fallbacks.generate_configs``."""
    parser = argparse.ArgumentParser(description="Generate machine-consumable LLM model configuration artifacts.")
    parser.add_argument(
        "--output-dir", default="configs", help="Directory to write generated artifacts (default: configs)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate(args.output_dir)


if __name__ == "__main__":
    main()
