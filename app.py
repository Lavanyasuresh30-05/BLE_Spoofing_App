import streamlit as st

# Page settings
st.set_page_config(
    page_title="BLE Spoofing Detection",
    page_icon="🔐",
    layout="wide"
)

# Title
st.title("🔐 BLE Spoofing Detection")
st.subheader("Ultra-Lightweight Quantized XGBoost for Smartwatches")

st.write(
    "This application is designed to detect suspicious BLE packets "
    "using an XGBoost-based machine learning pipeline."
)

# Team information
st.sidebar.title("Team 17")
st.sidebar.write("Lavanya & Sahana")

# Navigation
page = st.sidebar.radio(
    "Select Page",
    ["Home", "Model Information"]
)

# Home page
if page == "Home":

    st.header("🏠 Home")

    st.write(
        "Welcome to the Bluetooth Low Energy (BLE) Spoofing Detection App."
    )

    st.info(
        "This app will be connected to the trained XGBoost model "
        "and BLE dataset in the next step."
    )

    st.success("Phase 4 Part 2 storage validation: PASS ✅")


# Model information
elif page == "Model Information":

    st.header("📊 Phase 4 Part 2 Model Information")

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
    st.write("C Header: **BLE_XGBoost_INT4_inference.h**")
    st.write("Final Storage: **15.5527 KB**")
    st.write("Target Storage: **512 KB**")
    st.write("Target Capacity Used: **3.04%**")

    st.success(
        "STATUS: PASS — The INT4 representation fits below the 512 KB target."
    )
