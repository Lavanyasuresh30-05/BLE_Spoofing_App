import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------
# PAGE SETTINGS
# -------------------------------------------------
st.set_page_config(
    page_title="BLE Spoofing Detection",
    page_icon="🔐",
    layout="wide"
)

# -------------------------------------------------
# LOAD PHASE 4 MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("Phase4_XGBoost_Model.pkl")


model = load_model()

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("🔐 BLE Spoofing Detection")
st.subheader("Ultra-Lightweight Quantized XGBoost for Smartwatches")

st.write(
    "This application detects suspicious Bluetooth Low Energy "
    "(BLE) packets using a Phase 4 XGBoost machine learning model."
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("Team 17")
st.sidebar.write("Lavanya & Sahana")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Model Information",
        "BLE Spoofing Detection"
    ]
)

# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------
if page == "Home":

    st.header("🏠 Home")

    st.write(
        "Welcome to the Bluetooth Low Energy (BLE) "
        "Spoofing Detection App."
    )

    st.info(
        "The application uses the trained Phase 4 "
        "XGBoost model for BLE classification."
    )

    st.success("Phase 4 model loaded successfully ✅")


# -------------------------------------------------
# MODEL INFORMATION
# -------------------------------------------------
elif page == "Model Information":

    st.header("📊 Phase 4 Model Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("XGBoost Trees", "100")

    with col2:
        st.metric("Decision Nodes", "1,873")

    with col3:
        st.metric("INT4 Thresholds", "1,873")

    st.divider()

    st.subheader("INT4 Quantization Results")

    st.write("Packed INT4 bytes: **937 bytes**")
    st.write("C Header: **BLE_XGBoost_INT4_model.h**")
    st.write("Final Storage: **15.5527 KB**")
    st.write("Target Storage: **512 KB**")
    st.write("Target Capacity Used: **3.04%**")

    st.success(
        "STATUS: PASS — The INT4 representation fits "
        "below the 512 KB target."
    )


# -------------------------------------------------
# BLE SPOOFING DETECTION
# -------------------------------------------------
elif page == "BLE Spoofing Detection":

    st.header("🔍 BLE Spoofing Detection")

    st.write(
        "Upload a CSV file containing the BLE features "
        "used by the Phase 4 XGBoost model."
    )

    # These are the columns expected in the UPLOADED CSV.
    input_features = [
        "Timestamp_ms",
        "RSSI_dBm",
        "RSSI_Temporal_Variance",
        "Advertising_Interval_ms",
        "Channel",
        "Channel_Transition",
        "CONNECT_REQ_Timing_ms",
        "Packet_Length_bytes",
        "TxPower_dBm"
    ]

    # These are the FINAL features used by the Phase 4 model.
    model_features = [
        "RSSI_dBm",
        "RSSI_Temporal_Variance",
        "Advertising_Interval_ms",
        "Channel",
        "Channel_Transition",
        "CONNECT_REQ_Timing_ms",
        "Packet_Length_bytes",
        "TxPower_dBm",
        "Timestamp_relative_ms"
    ]

    st.subheader("Required BLE CSV Features")
    st.write(input_features)

    uploaded_file = st.file_uploader(
        "Upload BLE CSV Dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")

        st.write(
            f"Rows: **{df.shape[0]}** | "
            f"Columns: **{df.shape[1]}**"
        )

        st.dataframe(df.head())

        # -------------------------------------------------
        # CHECK REQUIRED INPUT COLUMNS
        # -------------------------------------------------
        missing_features = [
            feature
            for feature in input_features
            if feature not in df.columns
        ]

        if missing_features:

            st.error("Missing required features:")

            for feature in missing_features:
                st.write(f"- {feature}")

        else:

            # -------------------------------------------------
            # CREATE MODEL INPUT
            # -------------------------------------------------
            X_input = df[input_features].copy()

            # -------------------------------------------------
            # PHASE 4 TIMESTAMP PREPROCESSING
            # Same as the training notebook:
            #
            # Timestamp_relative_ms =
            # Timestamp_ms - minimum Timestamp_ms
            # -------------------------------------------------
            X_input["Timestamp_relative_ms"] = (
                pd.to_numeric(
                    X_input["Timestamp_ms"],
                    errors="coerce"
                )
                - pd.to_numeric(
                    X_input["Timestamp_ms"],
                    errors="coerce"
                ).min()
            )

            # Remove original Timestamp_ms
            X_input = X_input.drop(
                columns=["Timestamp_ms"]
            )

            # -------------------------------------------------
            # CONVERT ALL FEATURES TO NUMERIC
            # Same as Phase 4 training
            # -------------------------------------------------
            X_input = X_input.apply(
                pd.to_numeric,
                errors="coerce"
            )

            # -------------------------------------------------
            # REPLACE INFINITY WITH NaN
            # -------------------------------------------------
            X_input = X_input.replace(
                [np.inf, -np.inf],
                np.nan
            )

            # -------------------------------------------------
            # MEDIAN IMPUTATION
            # Same preprocessing used during Phase 4 training
            # -------------------------------------------------
            X_input = X_input.fillna(
                X_input.median()
            )

            # -------------------------------------------------
            # FORCE EXACT MODEL FEATURE ORDER
            # -------------------------------------------------
            X_input = X_input[model_features]
            
            # -------------------------------------------------
            # DEBUG INFORMATION
            # -------------------------------------------------
            st.subheader("Processed Model Input")

            st.write(
                "The CSV has been preprocessed using the "
                "same feature transformation used during Phase 4 training."
            )

            st.write("Final model features:")

            st.write(
                model_features
            )

            st.write("Data types:")

            st.write(
                X_input.dtypes
            )

            st.write("Remaining missing values:")

            st.write(
                X_input.isnull().sum()
            )

            # -------------------------------------------------
            # CHECK FOR REMAINING INVALID VALUES
            # -------------------------------------------------
            

            # -------------------------------------------------
            # DETECTION BUTTON
            # -------------------------------------------------
            if st.button("🚀 Detect BLE Spoofing"):

                try:

                    predictions = model.predict(
                        X_input
                    )

                    probabilities = (
                        model.predict_proba(X_input)
                    )

                    # -------------------------------------------------
                    # FIND PROBABILITY OF CLASS 1
                    # -------------------------------------------------
                    if hasattr(model, "classes_"):

                        classes = list(
                            model.classes_
                        )

                        if 1 in classes:

                            class_1_index = classes.index(1)

                            spoofing_probability = (
                                probabilities[:, class_1_index]
                            )

                        else:

                            spoofing_probability = (
                                probabilities[:, -1]
                            )

                    else:

                        spoofing_probability = (
                            probabilities[:, 1]
                        )

                    # -------------------------------------------------
                    # RESULTS
                    # -------------------------------------------------
                    result = df.copy()

                    result["Prediction"] = predictions

                    result["Spoofing_Probability"] = (
                        spoofing_probability
                    )

                    st.subheader(
                        "Detection Results"
                    )

                    normal_count = int(
                        (predictions == 0).sum()
                    )

                    spoofed_count = int(
                        (predictions == 1).sum()
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Normal Packets",
                            normal_count
                        )

                    with col2:

                        st.metric(
                            "Spoofed Packets",
                            spoofed_count
                        )

                    st.dataframe(
                        result,
                        use_container_width=True
                    )

                    st.success(
                        "BLE spoofing detection "
                        "completed successfully! ✅"
                    )

                except Exception as e:

                    st.error(
                        "Prediction failed."
                    )

                    st.exception(e)
