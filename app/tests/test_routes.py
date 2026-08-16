from io import BytesIO

from PIL import Image

from app import app

from pathlib import Path

from services.model_service import (
    predict_image
)

from unittest.mock import patch

def create_test_image():

    image = Image.new(
        "RGB",
        (
            128,
            128
        ),
        color=128
    )

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    buffer.seek(0)

    return buffer


def test_home_page():

    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200

    assert (
        b"Lung Nodule"
        in response.data
    )


def test_missing_upload():

    client = app.test_client()

    response = client.post(
        "/",
        data={},
        follow_redirects=True
    )

    assert response.status_code == 200


def test_invalid_extension():

    client = app.test_client()

    response = client.post(
        "/",
        data={
            "image": (
                BytesIO(
                    b"not-an-image"
                ),
                "sample.txt"
            )
        },
        content_type=
            "multipart/form-data",
        follow_redirects=True
    )

    assert response.status_code == 200

    assert (
        b"Unsupported file type"
        in response.data
    )


def test_corrupt_png():

    client = app.test_client()

    response = client.post(
        "/",
        data={
            "image": (
                BytesIO(
                    b"this-is-not-a-real-png"
                ),
                "broken.png"
            )
        },
        content_type=
            "multipart/form-data",
        follow_redirects=True
    )

    assert response.status_code == 200

    assert (
        b"valid image"
        in response.data
    )



TEST_IMAGE = (
    Path(__file__).parent
    / "fixtures"
    / "LIDC-IDRI-0044_Nodule_003_Malignant.png"
)


EXPECTED_PROBABILITY = (
   0.9188607335090637
)


def test_known_model_prediction():

    assert TEST_IMAGE.exists(), (
        f"Fixture image not found: {TEST_IMAGE}"
    )

    result = predict_image(
        TEST_IMAGE
    )

    print(
        "\nFixture:",
        TEST_IMAGE
    )

    print(
        "Predicted probability:",
        result["probability"]
    )

    print(
        "Expected probability:",
        EXPECTED_PROBABILITY
    )

    assert (
        result[
            "predicted_class"
        ]
        == "Malignant"
    )

    assert abs(
        result["probability"]
        - EXPECTED_PROBABILITY
    ) < 1e-4


# ============================================================
# Health Endpoint
# ============================================================

def test_health_endpoint():

    client = app.test_client()

    response = client.get(
        "/health"
    )

    assert (
        response.status_code
        == 200
    )

    data = response.get_json()

    assert (
        data["status"]
        == "ok"
    )

    assert (
        data["model"]
        == "DenseNet121"
    )

# ============================================================
# Oversized Upload
# ============================================================

def test_oversized_upload():

    client = app.test_client()

    oversized_data = (
        b"x"
        * (
            10 * 1024 * 1024
            + 1024
        )
    )

    response = client.post(
        "/",
        data={
            "image": (
                BytesIO(
                    oversized_data
                ),
                "large.png"
            )
        },
        content_type=
            "multipart/form-data",
        follow_redirects=True
    )

    assert (
        response.status_code
        == 200
    )

    assert (
        b"too large"
        in response.data.lower()
    )


@patch(
    "app.predict_image"
)
@patch(
    "app.validate_input_domain"
)
def test_unsupported_input_does_not_run_classifier(
    mock_domain_validation,
    mock_predict_image
):

    mock_domain_validation.return_value = {

        "probability":
            0.05,

        "threshold":
            0.968,

        "accepted":
            False,

        "status":
            "Unsupported"
    }


    client = app.test_client()


    test_image = create_test_image()


    response = client.post(
        "/",

        data={
            "image": (
                test_image,
                "test.png"
            )
        },

        content_type=
            "multipart/form-data"
    )


    assert (
        response.status_code
        == 200
    )


    assert (
        b"Unsupported Image"
        in response.data
    )


    mock_predict_image.assert_not_called()


@patch(
    "app.generate_explanation_safe"
)
@patch(
    "app.generate_gradcam"
)
@patch(
    "app.predict_image"
)
@patch(
    "app.validate_input_domain"
)
def test_supported_input_reaches_classifier(
    mock_domain_validation,
    mock_predict_image,
    mock_gradcam,
    mock_gemini
):

    mock_domain_validation.return_value = {

        "probability":
            0.99,

        "threshold":
            0.968,

        "accepted":
            True,

        "status":
            "Supported"
    }


    mock_predict_image.return_value = {

        "probability":
            0.90,

        "predicted_label":
            1,

        "predicted_class":
            "Malignant",

        "threshold":
            0.6690409183502197
    }


    mock_gradcam.return_value = None

    mock_gemini.return_value = (
        "Test explanation."
    )


    client = app.test_client()


    response = client.post(
        "/",

        data={
            "image": (
                create_test_image(),
                "test.png"
            )
        },

        content_type=
            "multipart/form-data"
    )


    assert (
        response.status_code
        == 200
    )


    mock_predict_image.assert_called_once()