import streamlit as st
import pandas as pd
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

    required_features = [
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

    st.subheader("Required BLE Features")
    st.write(required_features)

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

        missing_features = [
            feature
            for feature in required_features
            if feature not in df.columns
        ]

        if missing_features:

            st.error("Missing required features:")

            for feature in missing_features:
                st.write(f"- {feature}")

        else:

            X_input = df[required_features].copy()

            # Convert features to numbers
            for column in required_features:

                X_input[column] = pd.to_numeric(
                    X_input[column],
                    errors="coerce"
                )

            # Check for invalid values
            if X_input.isnull().any().any():

                st.error(
                    "Some feature values are missing "
                    "or are not numeric."
                )

                st.write(
                    X_input.isnull().sum()
                )

            else:

                if st.button("🚀 Detect BLE Spoofing"):

                    predictions = model.predict(
                        X_input
                    )

                    probabilities = (
                        model.predict_proba(X_input)
                    )

                    result = df.copy()

                    result["Prediction"] = predictions

                    result["Spoofing_Probability"] = (
                        probabilities[:, 1]
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

                    st.dataframe(result)

                    st.success(
                        "BLE spoofing detection "
                        "completed successfully! ✅"
                    )
