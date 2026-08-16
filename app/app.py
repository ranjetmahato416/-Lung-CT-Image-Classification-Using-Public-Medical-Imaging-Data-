from pathlib import Path
from uuid import uuid4

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash
)

from werkzeug.utils import secure_filename

from config import (
    UPLOAD_FOLDER,
    GRADCAM_FOLDER,
    MAX_CONTENT_LENGTH,
    ALLOWED_EXTENSIONS,
    FLASK_SECRET_KEY
)

from services.model_service import predict_image


from services.gemini_service import (
    generate_explanation_safe
)

from services.gradcam_service import (
    generate_gradcam,
    LAST_CONV_LAYER_NAME
)

from services.preprocessing import (
    validate_image_file
)

from services.file_service import (
    cleanup_old_files
)

from flask import url_for

from config import MODEL_PATH

from services.domain_validation_service import (
    validate_input_domain
)



app = Flask(__name__)

app.secret_key = (FLASK_SECRET_KEY)

app.config["UPLOAD_FOLDER"] = str(
    UPLOAD_FOLDER
)

app.config["GRADCAM_FOLDER"] = str(
    GRADCAM_FOLDER
)

app.config["MAX_CONTENT_LENGTH"] = (
    MAX_CONTENT_LENGTH
)


UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

GRADCAM_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


def allowed_file(filename):

    return (
        "."
        in filename
        and
        filename
        .rsplit(".", 1)[1]
        .lower()
        in ALLOWED_EXTENSIONS
    )


@app.route(
    "/",
    methods=[
        "GET",
        "POST"
    ]
)
def index():

    if request.method == "POST":

        # ------------------------------------------
        # Clean old temporary files
        # ------------------------------------------

        cleanup_old_files(
            UPLOAD_FOLDER,
            max_age_seconds=3600
        )

        cleanup_old_files(
            GRADCAM_FOLDER,
            max_age_seconds=3600
        )

        # ------------------------------------------
        # Validate uploaded field
        # ------------------------------------------

        if "image" not in request.files:

            flash(
                "No image was uploaded."
            )

            return redirect(
                request.url
            )

        file = request.files["image"]

        if file.filename == "":

            flash(
                "Please select an image."
            )

            return redirect(
                request.url
            )

        # ------------------------------------------
        # Validate extension
        # ------------------------------------------

        if not allowed_file(
            file.filename
        ):

            flash(
                "Unsupported file type. "
                "Please upload PNG, JPG or JPEG."
            )

            return redirect(
                request.url
            )

        # ------------------------------------------
        # Create safe unique filename
        # ------------------------------------------

        original_filename = (
            secure_filename(
                file.filename
            )
        )

        extension = (
            Path(
                original_filename
            )
            .suffix
            .lower()
        )

        unique_filename = (
            f"{uuid4().hex}"
            f"{extension}"
        )

        upload_path = (
            UPLOAD_FOLDER
            / unique_filename
        )

        file.save(
            upload_path
        )

        try:

            validate_image_file(
                upload_path
            )

        except ValueError:

            if upload_path.exists():
                upload_path.unlink()

            flash(
                "The uploaded file is not "
                "a valid image."
            )

            return redirect(
                request.url
            )

        # ------------------------------------------
        # Input-Domain Validation
        # ------------------------------------------

        domain_result = (
            validate_input_domain(
                upload_path
            )
        )


        if not domain_result[
            "accepted"
        ]:

            return render_template(
                "unsupported.html",

                filename=
                    unique_filename,

                original_filename=
                    original_filename,

                domain_probability=
                    domain_result[
                        "probability"
                    ],

                domain_threshold=
                    domain_result[
                        "threshold"
                    ]
            )

        # ------------------------------------------
        # DenseNet prediction
        # ------------------------------------------

        try:

            result = predict_image(
                upload_path
            )

            # --------------------------------------
            # Grad-CAM
            # --------------------------------------

            gradcam_filename = (
                f"gradcam_"
                f"{Path(unique_filename).stem}"
                f".png"
            )

            generate_gradcam(
                image_path=upload_path,
                predicted_label=result[
                    "predicted_label"
                ],
                output_filename=
                    gradcam_filename
            ),

            explanation = (
                generate_explanation_safe(
                    predicted_class=result[
                        "predicted_class"
                    ],

                    probability=result[
                        "probability"
                    ],

                    threshold=result[
                        "threshold"
                    ],

                    gradcam_layer=
                        LAST_CONV_LAYER_NAME
                )
            )

        except Exception as error:

            print(
                "Analysis error:",
                repr(error)
            )

            flash(
                "The uploaded image could not "
                "be analysed."
            )

            return redirect(
                request.url
            )

        # ------------------------------------------
        # Result page
        # ------------------------------------------

        return render_template(
            "result.html",

            filename=
                unique_filename,

            original_filename=
                original_filename,

            gradcam_filename=
                gradcam_filename,

            predicted_class=
                result[
                    "predicted_class"
                ],

            predicted_label=
                result[
                    "predicted_label"
                ],

            probability=
                result[
                    "probability"
                ],

            threshold=
                result[
                    "threshold"
                ],
             explanation=
                explanation
        )

    return render_template(
        "index.html"
    )

# ============================================================
# HTTP Error Handlers
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    flash(
        "The uploaded file is too large. "
        "Maximum allowed file size is 10 MB."
    )

    return redirect(
        url_for("index")
    )

# ============================================================
# Health Check
# ============================================================


@app.route("/health")
def health():

    return {
        "status": "ok",
        "application": "Lung Nodule Classification",
        "model": "DenseNet121",
        "model_file_available": MODEL_PATH.exists()
    }, 200


if __name__ == "__main__":

    app.run(
        debug=True
    )