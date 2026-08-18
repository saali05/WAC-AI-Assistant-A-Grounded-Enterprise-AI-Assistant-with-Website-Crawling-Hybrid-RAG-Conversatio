import pytest
from app.ai.pricing import calculate_cost, MODEL_PRICING


def test_free_tier_pricing_zero_cost():
    # Gemini 3.6 Flash free tier should yield 0.0
    cost_gemini = calculate_cost(
        model="gemini-3.6-flash",
        input_tokens=10000,
        output_tokens=5000,
        pricing_tier="free",
    )
    assert cost_gemini == 0.0

    # Gemini 3.1 Flash Live Preview free tier should yield 0.0
    cost_live = calculate_cost(
        model="gemini-3.1-flash-live-preview",
        input_tokens=5000,
        output_tokens=2000,
        pricing_tier="free",
    )
    assert cost_live == 0.0


def test_paid_tier_pricing():
    # Gemini 3.6 Flash paid tier: $1.50/1M input, $7.50/1M output
    # 100,000 input = 0.1M * 1.50 = 0.15
    # 50,000 output = 0.05M * 7.50 = 0.375
    # Total = 0.525
    cost_gemini = calculate_cost(
        model="gemini-3.6-flash",
        input_tokens=100000,
        output_tokens=50000,
        pricing_tier="paid",
    )
    assert cost_gemini == 0.525

    # Groq Llama 3.3 70B paid: $0.59/1M input, $0.79/1M output
    # 1,000,000 input = 0.59
    # 1,000,000 output = 0.79
    # Total = 1.38
    cost_groq = calculate_cost(
        model="llama-3.3-70b-versatile",
        input_tokens=1000000,
        output_tokens=1000000,
        pricing_tier="paid",
    )
    assert cost_groq == 1.38
