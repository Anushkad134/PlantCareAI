from openai import OpenAI
import os
import base64
import joblib
import pandas as pd
import streamlit as st


# =========================
# SETUP
# =========================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

model = joblib.load("plant_health_model.pkl")


st.set_page_config(
    page_title="Plant Health Analyzer",
    page_icon="🌿",
    layout="wide"
)


# =========================
# GREEN & WHITE UI
# =========================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #eef8f0,
        #ffffff,
        #e8f5eb
    );
}

.block-container {
    max-width: 1000px;
    padding-top: 40px;
    padding-bottom: 50px;
}

.stApp,
.stApp p,
.stApp span,
.stApp label,
.stApp li,
.stApp small,
.stApp strong {
    color: #234b31;
}

/* TITLE */

.main-title {
    text-align: center;
    color: #216e39 !important;
    font-size: 46px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #52745d !important;
    font-size: 18px;
    margin-bottom: 35px;
}

/* INFORMATION BOX */

.info {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 18px;
    border-left: 5px solid #4caf68;
    box-shadow: 0px 5px 20px rgba(40, 110, 60, 0.10);
    margin-bottom: 25px;
}

.info-title {
    color: #216e39 !important;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 8px;
}

.info-text {
    color: #52745d !important;
    font-size: 16px;
}

/* FILE UPLOADER */

[data-testid="stFileUploader"] {
    background-color: #ffffff !important;
    padding: 20px;
    border-radius: 18px;
    border: 2px dashed #91c79f;
}

[data-testid="stFileUploader"] label {
    color: #216e39 !important;
    font-weight: 600 !important;
}

[data-testid="stFileUploader"] section {
    color: #234b31 !important;
}

[data-testid="stFileUploader"] section * {
    color: #234b31 !important;
}

[data-testid="stFileUploader"] section div {
    color: #234b31 !important;
}

[data-testid="stFileUploader"] section span {
    color: #234b31 !important;
}

[data-testid="stFileUploader"] section small {
    color: #52745d !important;
}

[data-testid="stFileUploader"] button {
    color: #216e39 !important;
    background-color: #f1f8f3 !important;
    border: 1px solid #91c79f !important;
}

/* BUTTON */

.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 14px;
    border: none;
    background: linear-gradient(
        90deg,
        #4caf68,
        #216e39
    ) !important;
    color: #ffffff !important;
    font-size: 17px;
    font-weight: 700;
    box-shadow: 0px 6px 18px rgba(40, 110, 60, 0.20);
    transition: 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow:
        0px 9px 25px rgba(40, 110, 60, 0.30);
}

/* RESULT BOX */

.result {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 18px;
    border-left: 5px solid #4caf68;
    box-shadow: 0px 5px 20px rgba(40, 110, 60, 0.10);
    margin-top: 30px;
}

.result-title {
    color: #216e39 !important;
    font-size: 25px;
    font-weight: 700;
}

/* MARKDOWN */

.stMarkdown {
    color: #234b31 !important;
}

.stMarkdown p {
    color: #234b31 !important;
}

.stMarkdown li {
    color: #234b31 !important;
}

.stMarkdown strong {
    color: #216e39 !important;
}

.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3,
.stMarkdown h4,
.stMarkdown h5,
.stMarkdown h6 {
    color: #216e39 !important;
}

[data-testid="stMarkdownContainer"] {
    color: #234b31 !important;
}

[data-testid="stMarkdownContainer"] * {
    color: #234b31 !important;
}

[data-testid="stText"] {
    color: #234b31 !important;
}

/* IMAGE */

[data-testid="stImage"] {
    border-radius: 18px;
    overflow: hidden;
}

/* FOOTER */

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================
# HEADER
# =========================

st.markdown(
    '<div class="main-title">🌿 Plant Health Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered plant leaf disease detection & health analysis'
    '</div>',
    unsafe_allow_html=True
)


# =========================
# INFORMATION CARD
# =========================

st.markdown(
    '<div class="info">'
    '<div class="info-title">🌱 Analyze Your Plant</div>'
    '<div class="info-text">'
    'Choose between AI-powered leaf image analysis or '
    'plant health prediction based on environmental conditions.'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


# =========================
# CHOOSE ANALYSIS
# =========================

option = st.radio(
    "Choose Analysis Method",
    [
        "📷 Analyze Leaf Image",
        "📊 Predict Plant Health"
    ],
    horizontal=True
)


# =========================================================
# OPTION 1 — IMAGE ANALYSIS
# =========================================================

if option == "📷 Analyze Leaf Image":

    st.markdown(
        '<div class="info">'
        '<div class="info-title">📷 Leaf Image Analysis</div>'
        '<div class="info-text">'
        'Upload a clear image of a plant leaf and let AI analyze '
        'its health, possible diseases, visible symptoms, and '
        'preventive measures.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "📷 Upload a plant leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image_bytes = uploaded_file.read()

        st.image(
            image_bytes,
            caption="Uploaded Plant Leaf",
            width=400
        )

        image_base64 = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        if st.button("🌿 Analyze Plant Health"):

            response = client.responses.create(

                model="gpt-4.1-mini",

                input=[
                    {
                        "role": "user",

                        "content": [

                            {
                                "type": "input_text",

                                "text": """
Analyze this plant leaf image.

Identify:

1. Plant/crop if recognizable
2. Whether the leaf appears healthy or diseased
3. Possible disease
4. Visible symptoms
5. Possible causes
6. General preventive measures

If the image is unclear, explicitly state that the
diagnosis cannot be determined reliably.
"""
                            },

                            {
                                "type": "input_image",

                                "image_url":
                                f"data:image/jpeg;base64,{image_base64}"
                            }

                        ]
                    }
                ]
            )

            st.markdown(
                '<div class="result">'
                '<div class="result-title">'
                '🌱 Plant Health Analysis'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

            st.write(response.output_text)


# =========================================================
# OPTION 2 — MACHINE LEARNING PREDICTION
# =========================================================

else:

    st.markdown(
        '<div class="info">'
        '<div class="info-title">📊 Plant Health Prediction</div>'
        '<div class="info-text">'
        'Enter the plant characteristics and environmental '
        'conditions to predict its health status using the '
        'trained machine learning model.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


    # =========================
    # INPUTS
    # =========================

    col1, col2 = st.columns(2)


    with col1:

        plant_type = st.selectbox(
    "🌱 Plant Type",
    [
        "Tomato",
        "Potato",
        "Maize",
        "Rice",
        "Wheat",
        "Apple",
        "Banana",
        "Mango",
        "Orange",
        "Grapes",
        "Guava",
        "Papaya",
        "Pomegranate",
        "Lemon",
        "Peach",
        "Pear",
        "Plum",
        "Strawberry",
        "Watermelon",
        "Muskmelon",
        "Cucumber",
        "Pumpkin",
        "Carrot",
        "Radish",
        "Beetroot",
        "Spinach",
        "Lettuce",
        "Cabbage",
        "Cauliflower",
        "Broccoli",
        "Onion",
        "Garlic",
        "Peas",
        "Beans",
        "Chickpea",
        "Soybean",
        "Groundnut",
        "Cotton",
        "Sugarcane",
        "Sunflower",
        "Mustard",
        "Tea",
        "Coffee",
        "Coconut",
        "Papaya",
        "Aloe Vera",
        "Basil",
        "Mint",
        "Rose",
        "Jasmine",
        "Marigold",
        "Hibiscus",
        "Neem",
        "Tulsi",
        "Money Plant",
        "Snake Plant",
        "Peace Lily",
        "Arelia",
        "Fern",
        "Lavender",
        "Rosemary",
        "Chrysanthemum",
        "Orchid",
        "Bougainvillea",
        "Sunflower"
    ]
)
        

        habitat = st.selectbox(
            "🏡 Habitat",
            [
                "garden",
                "field",
                "greenhouse"
            ]
        )

        season = st.selectbox(
            "☀️ Season",
            [
                "summer",
                "winter",
                "spring",
                "monsoon"
            ]
        )

        temperature = st.number_input(
            "🌡️ Temperature (°C)",
            value=25.0
        )

        air_humidity = st.number_input(
            "💧 Air Humidity (%)",
            value=60.0
        )

        soil_humidity = st.number_input(
            "🌱 Soil Humidity (%)",
            value=50.0
        )


    with col2:

        ethylene = st.number_input(
            "Ethylene (ppm)",
            value=0.0
        )

        co2 = st.number_input(
            "CO₂ (ppm)",
            value=400.0
        )

        o2 = st.number_input(
            "O₂ (ppm)",
            value=210000.0
        )

        light = st.number_input(
            "☀️ Light Level",
            value=500.0
        )

        leaf_color = st.number_input(
            "🍃 Leaf Color Index",
            value=70.0
        )


    # =========================
    # PREDICTION
    # =========================

    if st.button("🌿 Predict Plant Health"):

        input_data = pd.DataFrame([{

            "plant_type": plant_type,

            "habitat": habitat,

            "season": season,

            "air_humidity_pct": air_humidity,

            "soil_humidity_pct": soil_humidity,

            "ethylene_ppm": ethylene,

            "co2_ppm": co2,

            "o2_ppm": o2,

            "ldr_on_plant": light,

            "leaf_color_index": leaf_color,

            "temperature_c": temperature

        }])


        prediction = model.predict(input_data)[0]


        # =========================
        # RESULT
        # =========================

        st.markdown(
            '<div class="result">'
            '<div class="result-title">'
            '🌱 Plant Health Prediction'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


        health = prediction.replace(
            "_", " "
        ).title()


        if prediction == "healthy":

            st.success(
                f"🌿 **Plant Health: {health}**"
            )

        elif prediction == "moderate_stress":

            st.warning(
                f"⚠️ **Plant Health: {health}**"
            )

        else:

            st.error(
                f"🚨 **Plant Health: {health}**"
            )


# =========================
# FOOTER
# =========================

st.markdown(
    """
    <br>
    <div style="
        text-align:center;
        color:#52745d;
        font-size:14px;
        padding-top:30px;
    ">
        🌿 Plant Health Analyzer • Python for Data Science
    </div>
    """,
    unsafe_allow_html=True
)