from unittest.mock import patch

from services.gemini_service import (
    build_explanation_prompt,
    generate_explanation_safe
)


# ============================================================
# Test 1 — Prompt Construction
# ============================================================

def test_gemini_prompt_contains_model_information():

    prompt = build_explanation_prompt(
        predicted_class="Malignant",
        probability=0.9921758770942688,
        threshold=0.6690409183502197,
        gradcam_layer="conv5_block16_2_conv"
    )

    assert "Malignant" in prompt

    assert "0.9922" in prompt

    assert "0.6690" in prompt

    assert (
        "conv5_block16_2_conv"
        in prompt
    )

    assert "Grad-CAM" in prompt


# ============================================================
# Test 2 — Gemini API Failure Fallback
# ============================================================

@patch(
    "services.gemini_service."
    "generate_explanation"
)
def test_gemini_failure_fallback(
    mock_generate
):

    mock_generate.side_effect = (
        RuntimeError(
            "Simulated Gemini API failure"
        )
    )

    result = generate_explanation_safe(
        predicted_class="Malignant",
        probability=0.90,
        threshold=0.6690,
        gradcam_layer=
            "conv5_block16_2_conv"
    )

    assert (
        "temporarily unavailable"
        in result
    )

    assert (
        "DenseNet121"
        in result
    )

    assert (
        "Grad-CAM"
        in result
    )


# ============================================================
# Test 3 — Verify Gemini Function Was Called
# ============================================================

@patch(
    "services.gemini_service."
    "generate_explanation"
)
def test_gemini_safe_wrapper_calls_service(
    mock_generate
):

    mock_generate.return_value = (
        "Test explanation generated successfully."
    )

    result = generate_explanation_safe(
        predicted_class="Benign",
        probability=0.25,
        threshold=0.6690,
        gradcam_layer=
            "conv5_block16_2_conv"
    )

    assert (
        result
        ==
        "Test explanation generated successfully."
    )

    mock_generate.assert_called_once()