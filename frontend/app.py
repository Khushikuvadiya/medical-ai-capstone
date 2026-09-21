import base64
import io
from pathlib import Path

import requests
import streamlit as st
from PIL import Image


# =========================================================
# SETTINGS
# =========================================================

API_BASE_URL = "http://127.0.0.1:8000"

PREDICT_URL = f"{API_BASE_URL}/predict"
EXPLAIN_URL = f"{API_BASE_URL}/explain"
COUNTERFACTUAL_URL = f"{API_BASE_URL}/counterfactual"
REVIEW_URL = f"{API_BASE_URL}/review"

OUTPUT_DIR = Path("outputs")


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Medical AI Decision Support",
    page_icon="🩻",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "explanation_result" not in st.session_state:
    st.session_state.explanation_result = None

if "counterfactual_result" not in st.session_state:
    st.session_state.counterfactual_result = None

if "current_file_id" not in st.session_state:
    st.session_state.current_file_id = None

if "review_status" not in st.session_state:
    st.session_state.review_status = None


# =========================================================
# TITLE
# =========================================================

st.title(
    "Counterfactual and Explainable "
    "Medical Imaging Decision Support"
)

st.warning(
    "Research prototype only. "
    "This system is not intended for clinical diagnosis "
    "or patient-care decisions."
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("System Information")

    st.write("Model: ResNet18")
    st.write("Task: Chest X-ray classification")
    st.write("Classes: NORMAL / PNEUMONIA")

    st.write(
        "Explainability: Grad-CAM, "
        "Integrated Gradients, Counterfactuals"
    )

    st.write("Prototype version: 0.5")

    st.divider()

    st.caption(
        "Academic research prototype. "
        "Not validated for clinical use."
    )


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def make_files(uploaded_file, image_bytes):

    return {
        "file": (
            uploaded_file.name,
            image_bytes,
            uploaded_file.type
        )
    }


def submit_review(
    decision,
    uploaded_file,
    prediction,
    normal_probability,
    pneumonia_probability
):

    payload = {
        "filename": uploaded_file.name,
        "prediction": prediction,
        "normal_probability": normal_probability,
        "pneumonia_probability": pneumonia_probability,
        "reviewer_decision": decision
    }

    try:

        response = requests.post(
            REVIEW_URL,
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        st.session_state.review_status = decision

        return True

    except requests.RequestException as error:

        st.error(
            "Reviewer decision could not be saved."
        )

        st.exception(error)

        return False


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    file_id = (
        uploaded_file.name,
        len(image_bytes)
    )


    # -----------------------------------------------------
    # RESET RESULTS FOR NEW IMAGE
    # -----------------------------------------------------

    if st.session_state.current_file_id != file_id:

        st.session_state.current_file_id = file_id

        st.session_state.prediction_result = None
        st.session_state.explanation_result = None
        st.session_state.counterfactual_result = None
        st.session_state.review_status = None


    original_image = Image.open(
        io.BytesIO(image_bytes)
    )

    st.subheader(
        "Uploaded X-ray"
    )

    st.image(
        original_image,
        caption=uploaded_file.name,
        width=500
    )


    # =====================================================
    # ANALYZE BUTTON
    # =====================================================

    if st.button(
        "Analyze X-ray",
        type="primary"
    ):

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        try:

            with st.spinner(
                "Running AI prediction..."
            ):

                prediction_response = requests.post(
                    PREDICT_URL,
                    files=make_files(
                        uploaded_file,
                        image_bytes
                    ),
                    timeout=60
                )

            prediction_response.raise_for_status()

            st.session_state.prediction_result = (
                prediction_response.json()
            )

        except requests.RequestException as error:

            st.error(
                "Prediction API request failed."
            )

            st.exception(error)

            st.stop()


        # -------------------------------------------------
        # GRAD-CAM
        # -------------------------------------------------

        try:

            with st.spinner(
                "Generating Grad-CAM explanation..."
            ):

                explanation_response = requests.post(
                    EXPLAIN_URL,
                    files=make_files(
                        uploaded_file,
                        image_bytes
                    ),
                    timeout=60
                )

            explanation_response.raise_for_status()

            st.session_state.explanation_result = (
                explanation_response.json()
            )

        except requests.RequestException as error:

            st.error(
                "Explainability API request failed."
            )

            st.exception(error)

            st.stop()


        st.session_state.counterfactual_result = None
        st.session_state.review_status = None


    # =====================================================
    # DISPLAY ANALYSIS RESULTS
    # =====================================================

    if (
        st.session_state.prediction_result is not None
        and
        st.session_state.explanation_result is not None
    ):

        prediction_result = (
            st.session_state.prediction_result
        )

        explanation_result = (
            st.session_state.explanation_result
        )

        prediction = prediction_result[
            "prediction"
        ]

        normal_probability = prediction_result[
            "normal_probability"
        ]

        pneumonia_probability = prediction_result[
            "pneumonia_probability"
        ]


        st.divider()

        st.header(
            "AI Analysis Results"
        )


        # -------------------------------------------------
        # PREDICTION METRICS
        # -------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Prediction",
                prediction
            )

        with col2:

            st.metric(
                "NORMAL probability",
                f"{normal_probability:.2%}"
            )

        with col3:

            st.metric(
                "PNEUMONIA probability",
                f"{pneumonia_probability:.2%}"
            )


        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        confidence = max(
            normal_probability,
            pneumonia_probability
        )

        if confidence < 0.60:

            st.warning(
                "Low-confidence prediction. "
                "The model is uncertain about this image."
            )

        elif confidence < 0.80:

            st.info(
                "Moderate-confidence prediction."
            )

        else:

            st.success(
                "High model confidence for this prediction."
            )


        # =================================================
        # GRAD-CAM
        # =================================================

        encoded_gradcam = explanation_result[
            "gradcam_image_base64"
        ]

        gradcam_bytes = base64.b64decode(
            encoded_gradcam
        )

        gradcam_image = Image.open(
            io.BytesIO(
                gradcam_bytes
            )
        )


        st.subheader(
            "Explainability Review"
        )

        left, right = st.columns(2)

        with left:

            st.image(
                original_image,
                caption="Original X-ray",
                width="stretch"
            )

        with right:

            st.image(
                gradcam_image,
                caption="Grad-CAM explanation",
                width="stretch"
            )


        st.info(
            "Grad-CAM highlights image regions that influenced "
            "the model prediction. Highlighted regions do not "
            "prove clinical causation or disease location."
        )


        # =================================================
        # COUNTERFACTUAL
        # =================================================

        st.divider()

        st.subheader(
            "Counterfactual Analysis"
        )

        st.write(
            "Generate an experimental counterfactual to examine "
            "what limited image changes can alter the model's "
            "prediction."
        )


        if st.button(
            "Generate Counterfactual",
            key="generate_counterfactual"
        ):

            try:

                with st.spinner(
                    "Generating counterfactual explanation..."
                ):

                    counterfactual_response = requests.post(
                        COUNTERFACTUAL_URL,
                        files=make_files(
                            uploaded_file,
                            image_bytes
                        ),
                        timeout=120
                    )

                counterfactual_response.raise_for_status()

                st.session_state.counterfactual_result = (
                    counterfactual_response.json()
                )

            except requests.RequestException as error:

                st.error(
                    "Counterfactual API request failed."
                )

                st.exception(error)


        # =================================================
        # DISPLAY COUNTERFACTUAL
        # =================================================

        if (
            st.session_state.counterfactual_result
            is not None
        ):

            cf = (
                st.session_state.counterfactual_result
            )


            encoded_counterfactual = cf[
                "counterfactual_image_base64"
            ]

            counterfactual_bytes = base64.b64decode(
                encoded_counterfactual
            )

            counterfactual_image = Image.open(
                io.BytesIO(
                    counterfactual_bytes
                )
            )


            encoded_difference = cf[
                "difference_image_base64"
            ]

            difference_bytes = base64.b64decode(
                encoded_difference
            )

            difference_image = Image.open(
                io.BytesIO(
                    difference_bytes
                )
            )


            st.subheader(
                "Counterfactual Result"
            )


            cf1, cf2, cf3 = st.columns(3)

            with cf1:

                st.image(
                    original_image,
                    caption="Original X-ray",
                    width="stretch"
                )

            with cf2:

                st.image(
                    counterfactual_image,
                    caption="Counterfactual X-ray",
                    width="stretch"
                )

            with cf3:

                st.image(
                    difference_image,
                    caption="Difference Map",
                    width="stretch"
                )


            st.subheader(
                "Prediction Change"
            )

            cf_metric1, cf_metric2 = st.columns(2)

            with cf_metric1:

                st.metric(
                    "Original Prediction",
                    cf[
                        "original_prediction"
                    ]
                )

            with cf_metric2:

                st.metric(
                    "Counterfactual Prediction",
                    cf[
                        "counterfactual_prediction"
                    ]
                )


            cf_metric3, cf_metric4 = st.columns(2)

            with cf_metric3:

                st.metric(
                    "Original PNEUMONIA Probability",
                    (
                        f"{cf['original_pneumonia_probability']:.2%}"
                    )
                )

            with cf_metric4:

                st.metric(
                    "Counterfactual NORMAL Probability",
                    (
                        f"{cf['counterfactual_normal_probability']:.2%}"
                    )
                )


            with st.expander(
                "Counterfactual technical metrics"
            ):

                st.write(
                    {
                        "mean_absolute_change":
                            cf[
                                "mean_absolute_change"
                            ],

                        "maximum_change":
                            cf[
                                "maximum_change"
                            ],

                        "fraction_changed_over_005":
                            cf[
                                "fraction_changed_over_005"
                            ]
                    }
                )


            st.warning(
                "The counterfactual is an experimental model "
                "interpretation artifact. It is not a clinically "
                "validated alternative X-ray and must not be used "
                "to infer treatment or biological causation."
            )


        # =================================================
        # REVIEWER DECISION
        # =================================================

        st.divider()

        st.subheader(
            "Reviewer Decision"
        )

        st.write(
            "Record the reviewer response to the AI prediction. "
            "The decision will be stored in the audit log."
        )


        review_col1, review_col2, review_col3 = (
            st.columns(3)
        )


        with review_col1:

            if st.button(
                "Agree with AI",
                use_container_width=True,
                key="agree_button"
            ):

                saved = submit_review(
                    "Agree with AI",
                    uploaded_file,
                    prediction,
                    normal_probability,
                    pneumonia_probability
                )

                if saved:

                    st.success(
                        "Reviewer decision saved: Agree with AI"
                    )


        with review_col2:

            if st.button(
                "Disagree with AI",
                use_container_width=True,
                key="disagree_button"
            ):

                saved = submit_review(
                    "Disagree with AI",
                    uploaded_file,
                    prediction,
                    normal_probability,
                    pneumonia_probability
                )

                if saved:

                    st.success(
                        "Reviewer decision saved: Disagree with AI"
                    )


        with review_col3:

            if st.button(
                "Needs further review",
                use_container_width=True,
                key="further_review_button"
            ):

                saved = submit_review(
                    "Needs further review",
                    uploaded_file,
                    prediction,
                    normal_probability,
                    pneumonia_probability
                )

                if saved:

                    st.success(
                        "Reviewer decision saved: "
                        "Needs further review"
                    )


        if st.session_state.review_status:

            st.info(
                "Current reviewer decision: "
                f"{st.session_state.review_status}"
            )


        # =================================================
        # CASE TECHNICAL DETAILS
        # =================================================

        with st.expander(
            "Case technical details"
        ):

            st.write(
                {
                    "filename":
                        uploaded_file.name,

                    "prediction":
                        prediction,

                    "normal_probability":
                        normal_probability,

                    "pneumonia_probability":
                        pneumonia_probability,

                    "confidence":
                        confidence,

                    "reviewer_decision":
                        st.session_state.review_status,

                    "research_use_only":
                        True
                }
            )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.divider()

st.header(
    "Model Performance"
)

st.write(
    "The following results summarize model performance "
    "on the held-out test dataset of 624 chest X-ray images."
)


# ---------------------------------------------------------
# CONFUSION MATRIX + ROC
# ---------------------------------------------------------

performance_col1, performance_col2 = (
    st.columns(2)
)


with performance_col1:

    st.subheader(
        "Confusion Matrix"
    )

    confusion_path = (
        OUTPUT_DIR
        / "confusion_matrix_baseline.png"
    )

    if confusion_path.exists():

        st.image(
            str(confusion_path),
            caption=(
                "Baseline ResNet18 Confusion Matrix"
            ),
            width="stretch"
        )

    else:

        st.warning(
            "Confusion matrix image not found."
        )


with performance_col2:

    st.subheader(
        "ROC Curve"
    )

    roc_path = (
        OUTPUT_DIR
        / "roc_curve_baseline.png"
    )

    if roc_path.exists():

        st.image(
            str(roc_path),
            caption=(
                "Baseline ResNet18 ROC Curve"
            ),
            width="stretch"
        )

    else:

        st.warning(
            "ROC curve image not found."
        )


# =========================================================
# BASELINE METRICS
# =========================================================

st.subheader(
    "Baseline Test Metrics"
)

metric1, metric2, metric3, metric4, metric5 = (
    st.columns(5)
)

with metric1:

    st.metric(
        "Accuracy",
        "82.85%"
    )

with metric2:

    st.metric(
        "Precision",
        "80.04%"
    )

with metric3:

    st.metric(
        "Recall",
        "96.67%"
    )

with metric4:

    st.metric(
        "F1 Score",
        "87.57%"
    )

with metric5:

    st.metric(
        "ROC-AUC",
        "94.34%"
    )


# =========================================================
# CONFUSION MATRIX EXPLANATION
# =========================================================

with st.expander(
    "How to read the confusion matrix"
):

    st.markdown(
        """
        - **140 NORMAL images** were correctly predicted as NORMAL.
        - **94 NORMAL images** were incorrectly predicted as PNEUMONIA.
        - **13 PNEUMONIA images** were incorrectly predicted as NORMAL.
        - **377 PNEUMONIA images** were correctly predicted as PNEUMONIA.

        The model has high pneumonia recall, but it also
        generates a noticeable number of false-positive
        pneumonia predictions.
        """
    )


# =========================================================
# ADVANCED TECHNICAL DETAILS
# =========================================================

with st.expander(
    "Advanced technical details"
):

    st.markdown(
        """
        ### Model Architecture

        **Architecture:** ResNet18  
        **Task:** Binary chest X-ray classification  
        **Input size:** 224 × 224 pixels  
        **Class mapping:** NORMAL = 0, PNEUMONIA = 1  
        **Training:** ImageNet pretrained transfer learning  

        ### Dataset

        **Original training images:** 5,216  
        **Training split:** 4,172  
        **Validation split:** 1,044  
        **Held-out test set:** 624  

        **Test NORMAL:** 234  
        **Test PNEUMONIA:** 390  

        ### Baseline Performance

        **Accuracy:** 82.85%  
        **Precision:** 80.04%  
        **Recall:** 96.67%  
        **F1 Score:** 87.57%  
        **ROC-AUC:** 94.34%  

        ### Calibration

        **Baseline Brier score:** 0.1172  
        **Baseline ECE:** 0.0679  

        **Temperature:** 0.8129  
        **Brier after scaling:** 0.1217  
        **ECE after scaling:** 0.0865  

        Temperature scaling did not improve held-out
        calibration and was therefore not retained as
        an improvement.

        ### Explainability Methods

        - Grad-CAM
        - Integrated Gradients
        - Deletion-based faithfulness testing
        - Counterfactual explanations

        ### Best Counterfactual Experiment

        **Method:** Low-resolution perturbation  
        **Target NORMAL probability:** 89.54%  
        **Mean absolute change:** 0.026228  
        **Maximum change:** 0.078849  
        **Fraction changed > 0.05:** 0.1150  

        Counterfactual images are experimental model
        interpretation artifacts and are not clinically
        validated medical images.

        ### Robustness Evaluation

        **Original accuracy:** 82.85%  
        **Darker:** 69.07%  
        **Brighter:** 87.50%  
        **Blurred:** 83.17%  
        **Gaussian noise:** 79.33%  
        **Low contrast:** 70.35%  

        The strongest degradation occurred under darker
        and low-contrast image conditions.

        ### Auditability

        Prediction, explainability, counterfactual activity,
        and reviewer decisions are recorded through the
        prototype audit logging system.

        ### Intended Use

        Academic research and demonstration only.

        This prototype is not validated for clinical
        diagnosis, treatment decisions, or patient care.
        """
    )