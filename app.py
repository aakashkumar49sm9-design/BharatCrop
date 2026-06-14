import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="BharatCrop",
    page_icon="🌾",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.stButton > button {
    width: 100%;
    height: 3.2em;
    font-size: 18px;
    font-weight: bold;
}

.big-font {
    font-size: 50px !important;
    font-weight: bold;
    color: #2E8B57;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD MODELS & DATASET
# =====================================

rf = joblib.load("crop_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

df = pd.read_csv("Crop_recommendation.csv")

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("🌾 BharatCrop")

st.sidebar.info("""
AI Powered Crop Recommendation System

Developed Using:
• Python
• Streamlit
• Scikit-Learn
• Machine Learning
• Data Analytics
""")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Crop Prediction",
        "Dataset Analytics",
        "Model Performance",
        "About"
    ]
)

# =====================================
# HOME PAGE
# =====================================

if page == "Home":

    st.markdown("""
    <div style='text-align:center;'>

    <h1 style='color:#2E8B57;font-size:60px;'>
    🌾 BharatCrop
    </h1>

    <h3>
    AI Powered Crop Recommendation System
    </h3>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background-color:#1E5631;
        padding:15px;
        border-radius:10px;
        text-align:center;
        margin-top:10px;
        margin-bottom:20px;
    ">

    <h2 style="color:white;">
    🚀 A PRODUCT BY TEAM ODESSEY 🚀
    </h2>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.write("""
    BharatCrop is an intelligent crop recommendation platform
    that helps farmers select the most suitable crop based on:

    • Nitrogen

    • Phosphorus

    • Potassium

    • Temperature

    • Humidity

    • pH

    • Rainfall

    The system uses Machine Learning techniques to provide
    accurate recommendations for better agricultural outcomes.
    """)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset Records", "2200")

    with col2:
        st.metric("Features", "7")

    with col3:
        st.metric("Crop Classes", "22")

    st.markdown("---")

    st.subheader("Key Features")

    st.write("""
    ✅ Crop Recommendation

    ✅ Soil Analysis

    ✅ Environmental Parameter Analysis

    ✅ Machine Learning Prediction

    ✅ Dataset Analytics Dashboard

    ✅ User Friendly Interface
    """)

# =====================================
# CROP PREDICTION PAGE
# =====================================

elif page == "Crop Prediction":

    st.title("🌱 Crop Prediction")

    st.write(
        "Enter soil and environmental parameters below."
    )

    col1, col2 = st.columns(2)

    with col1:

        N = st.number_input(
            "Nitrogen (N)",
            min_value=0.0
        )

        P = st.number_input(
            "Phosphorus (P)",
            min_value=0.0
        )

        K = st.number_input(
            "Potassium (K)",
            min_value=0.0
        )

        temp = st.number_input(
            "Temperature (°C)"
        )

    with col2:

        humidity = st.number_input(
            "Humidity (%)"
        )

        ph = st.number_input(
            "pH Value"
        )

        rainfall = st.number_input(
            "Rainfall (mm)"
        )

    if st.button("Predict Crop"):

        sample = np.array([
            [
                N,
                P,
                K,
                temp,
                humidity,
                ph,
                rainfall
            ]
        ])

        sample = scaler.transform(sample)

        prediction = rf.predict(sample)

        crop = encoder.inverse_transform(
            prediction.reshape(-1, 1)
        )

        st.balloons()

        st.success(
            f"🌾 Recommended Crop: {crop[0][0]}"
        )

        st.info(
            "Recommendation generated using the trained Random Forest model."
        )

# =====================================
# DATASET ANALYTICS PAGE
# =====================================

elif page == "Dataset Analytics":

    st.title("📊 Dataset Analytics")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Dataset Shape")

    st.write(df.shape)

    st.subheader("Statistical Summary")

    st.dataframe(df.describe())

    st.subheader("Crop Distribution")

    crop_counts = df["label"].value_counts()

    fig, ax = plt.subplots(figsize=(10,5))

    crop_counts.plot(
        kind="bar",
        ax=ax
    )

    plt.xticks(rotation=45)

    st.pyplot(fig)

# =====================================
# MODEL PERFORMANCE PAGE
# =====================================

elif page == "Model Performance":

    st.title("📈 Model Performance")

    model_data = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "ANN",
            "Transformer"
        ],
        "Accuracy": [
            0.95,
            0.99,
            0.90,
            0.71
        ]
    })

    st.dataframe(model_data)

    st.bar_chart(
        model_data.set_index("Model")
    )
    st.markdown("---")

    st.subheader("Feature Importance (Random Forest)")

    feature_names = [
        "Nitrogen",
        "Phosphorus",
        "Potassium",
        "Temperature",
        "Humidity",
        "pH",
        "Rainfall"
    ]

    importance = rf.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    st.dataframe(
        importance_df.sort_values(
            by="Importance",
            ascending=False
        )
    )

    st.bar_chart(
        importance_df.set_index("Feature")
    )
# =====================================
# ABOUT PAGE
# =====================================

elif page == "About":

    st.title("ℹ️ About BharatCrop")

    st.write("""
    BharatCrop is an AI Powered Crop Recommendation System
    developed to assist farmers and agricultural professionals
    in selecting suitable crops based on soil and climate data.
    """)

    st.subheader("Project Team")

    st.success("🚀 TEAM ODESSEY")

    st.subheader("Technologies Used")

    st.write("""
    • Python

    • NumPy

    • Pandas

    • Scikit-Learn

    • Streamlit

    • Machine Learning

    • Random Forest
    """)

