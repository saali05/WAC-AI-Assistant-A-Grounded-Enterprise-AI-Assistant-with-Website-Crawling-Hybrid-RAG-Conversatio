from typing import Any

from app.core.config import settings


# ============================================================
# MODEL PRICING
# ============================================================
#
# All prices are USD per 1,000,000 tokens.
#
# The providers consume this structure using:
#
#     MODEL_PRICING[model].get(...)
#
# Keep this as dictionaries because the provider layer already
# expects dictionary-style access.
# ============================================================

MODEL_PRICING: dict[str, dict[str, Any]] = {

    # --------------------------------------------------------
    # Gemini Text
    # --------------------------------------------------------

    settings.GEMINI_MODEL: {
        "input_per_1m": settings.GEMINI_INPUT_PRICE_PER_1M,
        "output_per_1m": settings.GEMINI_OUTPUT_PRICE_PER_1M,
        "context_limit": 1_048_576,
    },

    # --------------------------------------------------------
    # Groq
    # --------------------------------------------------------

    settings.GROQ_MODEL: {
        "input_per_1m": settings.GROQ_INPUT_PRICE_PER_1M,
        "output_per_1m": settings.GROQ_OUTPUT_PRICE_PER_1M,
        "context_limit": 131_072,
    },

    "llama-3.3-70b-versatile": {
        "input_per_1m": 0.59,
        "output_per_1m": 0.79,
        "context_limit": 131_072,
    },

    # --------------------------------------------------------
    # Gemini Live
    # --------------------------------------------------------

    settings.GEMINI_LIVE_MODEL: {
        "input_per_1m": settings.GEMINI_LIVE_TEXT_INPUT_PRICE_PER_1M,
        "output_per_1m": settings.GEMINI_LIVE_TEXT_OUTPUT_PRICE_PER_1M,
        "context_limit": 131_072,
    },
}


# ============================================================
# TOKEN COST
# ============================================================

def calculate_token_cost(
    input_tokens: int = 0,
    output_tokens: int = 0,
    input_price_per_1m: float = 0.0,
    output_price_per_1m: float = 0.0,
) -> float:
    """
    Calculate token-based API cost.

    Input:
        input_tokens
        output_tokens

    Pricing:
        USD per 1 million tokens
    """

    input_tokens = max(int(input_tokens or 0), 0)
    output_tokens = max(int(output_tokens or 0), 0)

    input_cost = (
        input_tokens / 1_000_000
    ) * input_price_per_1m

    output_cost = (
        output_tokens / 1_000_000
    ) * output_price_per_1m

    return round(
        input_cost + output_cost,
        10,
    )


# ============================================================
# MODEL COST
# ============================================================

def calculate_model_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> float:
    """
    Calculate the exact configured cost for a model.
    """

    model_spec = MODEL_PRICING.get(model)

    if not model_spec:
        if "llama" in model.lower() or "groq" in model.lower():
            model_spec = {
                "input_per_1m": settings.GROQ_INPUT_PRICE_PER_1M,
                "output_per_1m": settings.GROQ_OUTPUT_PRICE_PER_1M,
            }
        elif "gemini" in model.lower():
            model_spec = {
                "input_per_1m": settings.GEMINI_INPUT_PRICE_PER_1M,
                "output_per_1m": settings.GEMINI_OUTPUT_PRICE_PER_1M,
            }
        else:
            return 0.0

    return calculate_token_cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_price_per_1m=model_spec.get(
            "input_per_1m",
            0.0,
        ),
        output_price_per_1m=model_spec.get(
            "output_per_1m",
            0.0,
        ),
    )


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def calculate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    pricing_tier: str = "paid",
    audio_input_seconds: float | None = None,
    audio_output_seconds: float | None = None,
    **kwargs,
) -> float:
    """
    Backward-compatible cost calculation.

    Existing provider code and tests can continue calling:

        calculate_cost(
            model,
            input_tokens,
            output_tokens,
            pricing_tier=...,
        )
    """
    if pricing_tier == "free":
        return 0.0

    return calculate_model_cost(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


# ============================================================
# GEMINI LIVE TEXT COST
# ============================================================

def calculate_live_text_cost(
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> float:

    return calculate_token_cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_price_per_1m=(
            settings.GEMINI_LIVE_TEXT_INPUT_PRICE_PER_1M
        ),
        output_price_per_1m=(
            settings.GEMINI_LIVE_TEXT_OUTPUT_PRICE_PER_1M
        ),
    )


# ============================================================
# GEMINI LIVE AUDIO COST
# ============================================================

def calculate_live_audio_cost(
    input_audio_tokens: int = 0,
    output_audio_tokens: int = 0,
) -> float:

    return calculate_token_cost(
        input_tokens=input_audio_tokens,
        output_tokens=output_audio_tokens,
        input_price_per_1m=(
            settings.GEMINI_LIVE_AUDIO_INPUT_PRICE_PER_1M
        ),
        output_price_per_1m=(
            settings.GEMINI_LIVE_AUDIO_OUTPUT_PRICE_PER_1M
        ),
    )


# ============================================================
# GET MODEL PRICING
# ============================================================

def get_model_pricing(
    model: str,
) -> dict[str, Any] | None:

    return MODEL_PRICING.get(model)