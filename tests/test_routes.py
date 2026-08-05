from io import BytesIO

from PIL import Image

from app import app

from pathlib import Path

from services.model_service import (
    predict_image
)

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
    / "LIDC-IDRI-0050_Nodule_001.png"
)


EXPECTED_PROBABILITY = (
    0.9631621241569519
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