from pathlib import Path

from services.domain_validation_service import (
    validate_input_domain
)


TEST_CT_IMAGE = (
    Path(__file__).parent
    / "fixtures"
    / "LIDC-IDRI-0044_Nodule_003_Malignant.png"
)


def test_supported_ct_domain():

    result = validate_input_domain(
        TEST_CT_IMAGE
    )

    assert (
        result[
            "accepted"
        ]
        is True
    )

    assert (
        result[
            "probability"
        ]
        >=
        result[
            "threshold"
        ]
    )

TEST_XRAY_IMAGE = (
    Path(__file__).parent
    / "fixtures"
    / "00000061_004.png"
)


def test_xray_is_rejected():

    result = validate_input_domain(
        TEST_XRAY_IMAGE
    )

    assert (
        result[
            "accepted"
        ]
        is False
    )

    assert (
        result[
            "probability"
        ]
        <
        result[
            "threshold"
        ]
    )