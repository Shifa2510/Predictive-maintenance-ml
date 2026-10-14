from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance ML",
    page_icon="⚙️",
    layout="wide"
)


# --------------------------------------------------
# 2. Load the Saved Model
# --------------------------------------------------

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "random_forest_model.joblib"
)

FAILURE_THRESHOLD = 0.40


@st.cache_resource
def load_model():
    """Load the trained machine-learning pipeline."""
    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# 3. Application Header
# --------------------------------------------------

st.title("⚙️ Predictive Maintenance Dashboard")

st.markdown(
    """
    ### Machine Failure Prediction

    Enter machine sensor measurements to estimate the probability
    of machine failure using a trained Random Forest classifier.

    This application demonstrates how machine-learning models
    can support industrial predictive maintenance.
    """
)

st.info(
    "This is a demonstration using the AI4I 2020 Predictive "
    "Maintenance Dataset. Predictions are not a substitute for "
    "professional equipment inspection."
)


# --------------------------------------------------
# 4. Navigation
# --------------------------------------------------

prediction_tab, evaluation_tab, about_tab = st.tabs(
    [
        "🔮 Failure Prediction",
        "📊 Model Evaluation",
        "ℹ️ About Project"
    ]
)


# --------------------------------------------------
# 5. Prediction Interface
# --------------------------------------------------

with prediction_tab:

    st.header("Enter Machine Sensor Measurements")

    st.write(
        "Adjust the values below to estimate the likelihood "
        "of machine failure."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            machine_type = st.selectbox(
                "Machine Type",
                options=["L", "M", "H"],
                help=(
                    "L, M and H represent the three product "
                    "quality categories in the dataset."
                )
            )

            air_temperature = st.number_input(
                "Air Temperature (K)",
                min_value=250.0,
                max_value=350.0,
                value=300.0,
                step=0.1
            )

            process_temperature = st.number_input(
                "Process Temperature (K)",
                min_value=250.0,
                max_value=400.0,
                value=310.0,
                step=0.1
            )

        with col2:

            rotational_speed = st.number_input(
                "Rotational Speed (rpm)",
                min_value=0,
                max_value=10000,
                value=1500,
                step=10
            )

            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0,
                max_value=200.0,
                value=40.0,
                step=0.5
            )

            tool_wear = st.number_input(
                "Tool Wear (min)",
                min_value=0,
                max_value=3000,
                value=100,
                step=1
            )

        submitted = st.form_submit_button(
            "Predict Machine Failure",
            use_container_width=True
        )

    if submitted:

        # Prepare the input using the exact feature names
        # expected by the trained model.

        input_data = pd.DataFrame(
            {
                "Type": [machine_type],
                "Air temperature [K]": [air_temperature],
                "Process temperature [K]": [
                    process_temperature
                ],
                "Rotational speed [rpm]": [
                    rotational_speed
                ],
                "Torque [Nm]": [torque],
                "Tool wear [min]": [tool_wear]
            }
        )

        try:

            model = load_model()

            # Obtain the estimated probability of failure.
            failure_probability = model.predict_proba(
                input_data
            )[0][1]

            # Apply the threshold selected during validation.
            prediction = int(
                failure_probability >= FAILURE_THRESHOLD
            )

            st.divider()

            st.subheader("Prediction Results")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.metric(
                    label="Estimated Failure Probability",
                    value=f"{failure_probability:.2%}"
                )

            with result_col2:

                st.metric(
                    label="Classification Threshold",
                    value=f"{FAILURE_THRESHOLD:.0%}"
                )

            st.progress(
                float(failure_probability),
                text="Estimated probability of machine failure"
            )

            if prediction == 1:

                st.error(
                    "⚠️ **Potential Machine Failure Detected**\n\n"
                    "The predicted probability meets or exceeds "
                    "the configured classification threshold. "
                    "Consider further inspection by a qualified "
                    "maintenance professional."
                )

            else:

                st.success(
                    "✅ **No Failure Predicted at the Selected "
                    "Threshold**\n\n"
                    "The predicted probability is below the "
                    "configured classification threshold. "
                    "This does not guarantee that the machine "
                    "is operating safely."
                )

            st.caption(
                "The probability is the model's estimated "
                "failure probability, not a guarantee of the "
                "machine's actual condition."
            )

            with st.expander("View Submitted Sensor Measurements"):

                st.dataframe(
                    input_data,
                    use_container_width=True,
                    hide_index=True
                )

        except FileNotFoundError:

            st.error(
                "The saved model could not be found. "
                "Make sure models/random_forest_model.joblib "
                "exists in your project directory."
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# --------------------------------------------------
# 6. Model Evaluation Dashboard
# --------------------------------------------------

with evaluation_tab:

    st.header("Random Forest Model Evaluation")

    st.write(
        "The following results were obtained from the "
        "held-out test set using a classification threshold "
        "of 0.40."
    )

    st.caption(
        "These are the previously recorded evaluation results. "
        "They are displayed as reported results rather than "
        "being recalculated by this dashboard."
    )

    # Recorded evaluation metrics

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Accuracy",
        "98.05%"
    )

    metric2.metric(
        "Precision",
        "83.72%"
    )

    metric3.metric(
        "Recall",
        "52.94%"
    )

    metric4.metric(
        "F1-score",
        "64.86%"
    )

    st.divider()

    st.subheader("Confusion Matrix")

    # Rows represent actual labels.
    # Columns represent predicted labels.

    confusion_matrix = pd.DataFrame(
        [
            [1925, 7],
            [32, 36]
        ],
        index=[
            "Actual: No Failure",
            "Actual: Failure"
        ],
        columns=[
            "Predicted: No Failure",
            "Predicted: Failure"
        ]
    )

    st.dataframe(
        confusion_matrix,
        use_container_width=True
    )

    st.markdown(
        """
        **How to interpret these results**

        - **1,925 true negatives:** Non-failure records
          correctly classified as non-failures.
        - **7 false positives:** Non-failure records
          incorrectly classified as failures.
        - **32 false negatives:** Actual failures that the
          model did not detect.
        - **36 true positives:** Actual failures correctly
          detected by the model.
        """
    )

    st.warning(
        "The model missed 32 of the 68 actual failures in the "
        "test set. Improving recall is an important direction "
        "for future work."
    )

    st.subheader("Understanding the Evaluation Metrics")

    with st.expander("Accuracy"):

        st.write(
            "The proportion of all test records classified "
            "correctly. Accuracy can be misleading when failures "
            "are rare."
        )

    with st.expander("Precision"):

        st.write(
            "Among records predicted as failures, the proportion "
            "that were actual failures."
        )

    with st.expander("Recall"):

        st.write(
            "Among all actual failures, the proportion the model "
            "successfully detected."
        )

    with st.expander("F1-score"):

        st.write(
            "The harmonic mean of precision and recall. It "
            "balances both metrics into a single score."
        )


# --------------------------------------------------
# 7. Project Information
# --------------------------------------------------

with about_tab:

    st.header("About This Project")

    st.markdown(
        """
        **Objective:** Explore industrial machine sensor data
        and predict potential machine failures.

        **Dataset:** AI4I 2020 Predictive Maintenance Dataset
        from the UCI Machine Learning Repository.

        **Model:** Random Forest classifier.

        **Libraries:** Python, Pandas, Scikit-learn,
        Joblib and Streamlit.

        **Workflow:**

        1. Data understanding
        2. Exploratory data analysis
        3. Feature selection and preprocessing
        4. Model training
        5. Validation-based threshold selection
        6. Test-set evaluation
        7. Interactive predictions

        **Important limitation:** This is an educational
        demonstration. The model has not been validated for
        real-world industrial deployment.
        """
    )

    st.markdown(
        """
        ---
        **Predictive Maintenance ML | Machine Learning Portfolio Project**
        """
    )