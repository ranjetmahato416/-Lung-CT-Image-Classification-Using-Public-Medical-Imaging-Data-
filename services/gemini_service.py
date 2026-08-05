from google import genai

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL
)


_client = None


def get_gemini_client():
    """
    Create and cache the Gemini API client.
    """

    global _client

    if not GEMINI_API_KEY:

        raise RuntimeError(
            "GEMINI_API_KEY is missing. "
            "Add it to the .env file."
        )

    if _client is None:

        _client = genai.Client(
            api_key=GEMINI_API_KEY
        )

    return _client


def build_explanation_prompt(
    predicted_class,
    probability,
    threshold,
    gradcam_layer
):
    """
    Build a tightly controlled prompt.

    Gemini explains the DenseNet result.
    It must not independently diagnose the CT image.
    """

    probability_percent = (
        probability * 100
    )

    threshold_percent = (
        threshold * 100
    )

    return f"""
You are explaining the output of an academic machine-learning
research prototype for lung nodule classification.

The prediction has already been made by a fine-tuned DenseNet121
model. Do not make an independent medical diagnosis and do not
override the model result.

Model information:
- Architecture: Fine-Tuned DenseNet121
- Predicted class: {predicted_class}
- Model output score: {probability:.4f}
- Model output percentage: {probability_percent:.2f}%
- Validation-selected decision threshold: {threshold:.4f}
- Threshold percentage: {threshold_percent:.2f}%
- Explainability method: Grad-CAM
- Grad-CAM layer: {gradcam_layer}

Explain this result in plain language for a user of an academic
research prototype.

Requirements:
1. Explain why the predicted class was produced in relation to the
   model score and decision threshold.
2. Explain that Grad-CAM highlights image regions that influenced
   the model's predicted class.
3. State that Grad-CAM does not show exact tumour boundaries.
4. Make clear that the model output is not a clinical diagnosis.
5. Do not claim that cancer is definitely present or absent.
6. Do not recommend treatment or medical procedures.
7. Keep the explanation concise, around 120-180 words.
8. Use clear paragraphs rather than a long list.
"""


def generate_explanation(
    predicted_class,
    probability,
    threshold,
    gradcam_layer
):
    """
    Generate a plain-language explanation using Gemini.
    """

    client = get_gemini_client()

    prompt = build_explanation_prompt(
        predicted_class=predicted_class,
        probability=probability,
        threshold=threshold,
        gradcam_layer=gradcam_layer
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    if not response.text:

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return response.text.strip()

def generate_explanation_safe(
    predicted_class,
    probability,
    threshold,
    gradcam_layer
):
    """
    Generate Gemini explanation without allowing
    API failure to break the main ML application.
    """

    try:

        return generate_explanation(
            predicted_class=predicted_class,
            probability=probability,
            threshold=threshold,
            gradcam_layer=gradcam_layer
        )

    except Exception as error:

        print(
            "Gemini explanation error:",
            repr(error)
        )

        return (
            "An AI-generated textual explanation "
            "is temporarily unavailable. "
            "The DenseNet121 prediction and Grad-CAM "
            "visualization were generated successfully."
        )