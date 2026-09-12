```python
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="Lost & Found AI",
    page_icon="🔎",
    layout="centered"
)


# ============================================================
# DESIGN
# ============================================================

st.markdown("""
<style>

    /* Background */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    }

    /* Main container */
    .block-container {
        max-width: 850px;
        padding-top: 45px;
        padding-bottom: 50px;
    }

    /* Header */
    .title {
        text-align: center;
        font-size: 44px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 17px;
        margin-bottom: 35px;
    }

    /* Upload box */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.8);
        border-radius: 20px;
        padding: 10px;
        border: 1px solid #e2e8f0;
    }

    /* Result */
    .result {
        background: white;
        border-radius: 22px;
        padding: 30px;
        margin-top: 30px;
        text-align: center;
        box-shadow: 0 10px 35px rgba(15,23,42,0.08);
        border: 1px solid #e2e8f0;
    }

    .result-small {
        color: #64748b;
        font-size: 16px;
        margin-bottom: 7px;
    }

    .result-name {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .result-percent {
        color: #64748b;
        font-size: 18px;
    }

    /* Category */
    .category {
        background: white;
        border-radius: 14px;
        padding: 14px 17px;
        margin-top: 10px;
        border: 1px solid #e2e8f0;
    }

    .category-name {
        font-weight: 600;
    }

    .category-percent {
        float: right;
        color: #64748b;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 13px;
        margin-top: 45px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title">🔎 Lost & Found AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Lade ein Foto hoch und die KI erkennt, was darauf zu sehen ist.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LABELS
# ============================================================

labels = [
    "Kleidungsstücke",
    "Schulsachen",
    "Trinkflasche",
    "Brotdose",
    "Regenschirm",
    "Schlüssel",
    "Kopfhörer",
    "Powerbank/Ladekabel",
    "Brille",
    "Geldtasche",
    "Taschenrechner"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "keras_model.h5",
        compile=False
    )


try:
    model = load_model()

except Exception as e:
    st.error("❌ Das KI-Modell konnte nicht geladen werden.")
    st.stop()


# ============================================================
# UPLOAD IMAGE
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Foto hochladen",
    type=["jpg", "jpeg", "png", "webp"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Dein Bild",
        use_container_width=True
    )

    st.write("")

    if st.button(
        "🔍 Bild analysieren",
        use_container_width=True
    ):

        with st.spinner("Die KI analysiert das Bild..."):

            # ------------------------------------------------
            # IMAGE PREPROCESSING
            # ------------------------------------------------

            image_resized = image.resize((224, 224))

            image_array = np.asarray(
                image_resized
            ).astype(np.float32)

            # Teachable Machines normalization
            image_array = (image_array / 127.5) - 1

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

            # ------------------------------------------------
            # FIND BEST RESULT
            # ------------------------------------------------

            best_index = int(
                np.argmax(predictions)
            )

            best_label = labels[best_index]

            best_probability = (
                float(predictions[best_index]) * 100
            )

        # ====================================================
        # MAIN RESULT
        # ====================================================

        st.markdown(
            f"""
            <div class="result">

                <div class="result-small">
                    Die KI erkennt wahrscheinlich:
                </div>

                <div class="result-name">
                    {best_label}
                </div>

                <div class="result-percent">
                    {best_probability:.1f}% Wahrscheinlichkeit
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # ALL RESULTS
        # ====================================================

        st.markdown("### 📊 Alle Kategorien")

        results = sorted(
            zip(labels, predictions),
            key=lambda x: x[1],
            reverse=True
        )

        for label, probability in results:

            percentage = float(probability) * 100

            st.markdown(
                f"""
                <div class="category">

                    <span class="category-name">
                        {label}
                    </span>

                    <span class="category-percent">
                        {percentage:.1f}%
                    </span>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                min(float(probability), 1.0)
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        AI Lost & Found • Powered by TensorFlow & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
```
