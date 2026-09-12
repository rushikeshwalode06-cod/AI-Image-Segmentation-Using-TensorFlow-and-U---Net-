import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json
import h5py


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Image Segmentation Studio",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PREMIUM DESIGN CSS — DESIGN ONLY
# ============================================================

st.html("""
<style>

/* ============================================================
   MAIN BACKGROUND
   ============================================================ */

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(
            circle at 5% 5%,
            rgba(99,102,241,0.14),
            transparent 28%
        ),
        radial-gradient(
            circle at 95% 10%,
            rgba(236,72,153,0.13),
            transparent 30%
        ),
        radial-gradient(
            circle at 50% 100%,
            rgba(14,165,233,0.08),
            transparent 30%
        ),
        #f7f8fc;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            rgba(255,255,255,0.98),
            rgba(248,250,252,0.96)
        );

    border-right: 1px solid rgba(148,163,184,0.18);
}

.sidebar-title {
    font-size: 22px;
    font-weight: 900;
    color: #172033;
    margin-bottom: 10px;
}

.sidebar-card {
    padding: 18px;
    border-radius: 20px;

    background:
        linear-gradient(
            135deg,
            #eef2ff,
            #fdf2f8
        );

    border: 1px solid #e2e8f0;

    box-shadow:
        0 10px 28px rgba(15,23,42,0.06);
}


/* ============================================================
   HERO SECTION
   ============================================================ */

.hero {
    position: relative;
    overflow: hidden;

    padding: 55px 30px 48px;

    margin: 5px 0 30px;

    border-radius: 32px;

    background:
        radial-gradient(
            circle at 15% 20%,
            rgba(255,255,255,0.20),
            transparent 23%
        ),
        radial-gradient(
            circle at 85% 80%,
            rgba(255,255,255,0.15),
            transparent 25%
        ),
        linear-gradient(
            135deg,
            #4338ca,
            #7c3aed,
            #db2777
        );

    color: white;

    text-align: center;

    box-shadow:
        0 25px 65px rgba(79,70,229,0.28);

    border:
        1px solid rgba(255,255,255,0.18);
}

.hero::before {
    content: "";

    position: absolute;

    width: 210px;
    height: 210px;

    left: -90px;
    top: -100px;

    border-radius: 50%;

    background:
        rgba(255,255,255,0.08);
}

.hero::after {
    content: "";

    position: absolute;

    width: 250px;
    height: 250px;

    right: -110px;
    bottom: -130px;

    border-radius: 50%;

    background:
        rgba(255,255,255,0.08);
}

.hero-icon {
    font-size: 70px;
    line-height: 1;
    margin-bottom: 15px;

    filter:
        drop-shadow(
            0 8px 18px rgba(0,0,0,0.18)
        );
}

.hero-title {
    font-size: clamp(34px, 4vw, 52px);

    font-weight: 900;

    letter-spacing: -1.5px;

    line-height: 1.1;
}

.hero-subtitle {
    font-size: 22px;

    font-weight: 700;

    margin-top: 12px;
}

.hero-description {
    max-width: 800px;

    margin: 17px auto 0;

    font-size: 16px;

    line-height: 1.6;

    opacity: 0.93;
}

.badge {
    display: inline-block;

    padding: 9px 16px;

    margin: 18px 4px 0;

    border-radius: 999px;

    background:
        rgba(255,255,255,0.14);

    border:
        1px solid rgba(255,255,255,0.28);

    font-size: 13px;

    font-weight: 700;

    backdrop-filter: blur(10px);
}


/* ============================================================
   SECTION TITLE
   ============================================================ */

.section-title {
    font-size: 27px;

    font-weight: 900;

    margin:
        30px 0 15px;

    color: #172033;

    letter-spacing: -0.5px;
}

.section-subtitle {
    color: #64748b;

    font-size: 14px;

    margin-top: -7px;

    margin-bottom: 18px;
}


/* ============================================================
   GENERAL CARD
   ============================================================ */

.info-card {

    padding: 28px;

    border-radius: 25px;

    background:
        rgba(255,255,255,0.90);

    border:
        1px solid rgba(226,232,240,0.95);

    box-shadow:
        0 14px 38px rgba(15,23,42,0.07);

    backdrop-filter: blur(15px);
}


/* ============================================================
   RESULT HEADER
   ============================================================ */

.result-banner {

    display: flex;

    align-items: center;

    justify-content: space-between;

    gap: 20px;

    padding: 21px 24px;

    margin: 5px 0 22px;

    border-radius: 22px;

    background:
        linear-gradient(
            135deg,
            #eef2ff,
            #fdf2f8
        );

    border:
        1px solid #e2e8f0;

    box-shadow:
        0 10px 28px rgba(15,23,42,0.06);
}

.result-title {

    font-size: 21px;

    font-weight: 900;

    color: #172033;
}

.result-meta {

    color: #64748b;

    font-size: 13px;

    margin-top: 4px;
}

.result-status {

    color: #15803d;

    font-size: 13px;

    font-weight: 900;

    white-space: nowrap;
}


/* ============================================================
   METRIC CARDS
   ============================================================ */

.metric-card {

    min-height: 135px;

    padding: 20px 15px;

    border-radius: 22px;

    background:
        rgba(255,255,255,0.96);

    border:
        1px solid #e5e7eb;

    text-align: center;

    box-shadow:
        0 10px 30px rgba(15,23,42,0.065);

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease;
}

.metric-card:hover {

    transform:
        translateY(-6px);

    box-shadow:
        0 18px 38px rgba(79,70,229,0.15);
}

.metric-icon {

    font-size: 25px;

    margin-bottom: 5px;
}

.metric-value {

    font-size: 28px;

    font-weight: 900;

    color: #4f46e5;

    line-height: 1.1;
}

.metric-label {

    font-size: 13px;

    color: #64748b;

    margin-top: 8px;

    font-weight: 700;
}


/* ============================================================
   CONFIDENCE CARDS
   ============================================================ */

.confidence-card {

    padding: 23px;

    border-radius: 23px;

    background:
        linear-gradient(
            135deg,
            #ffffff,
            #f8fafc
        );

    border:
        1px solid #e2e8f0;

    box-shadow:
        0 10px 30px rgba(15,23,42,0.06);
}

.confidence-label {

    color: #64748b;

    font-size: 12px;

    font-weight: 800;

    letter-spacing: 0.5px;
}

.confidence-number {

    font-size: 34px;

    font-weight: 900;

    color: #7c3aed;

    margin:
        5px 0;
}

.mini-info {

    padding: 14px 16px;

    border-radius: 16px;

    background:
        #f8fafc;

    border:
        1px solid #e2e8f0;

    color: #475569;

    font-size: 13px;
}


/* ============================================================
   IMAGE CARDS
   ============================================================ */

.image-card {

    padding: 14px;

    border-radius: 23px;

    background:
        rgba(255,255,255,0.96);

    border:
        1px solid #e2e8f0;

    box-shadow:
        0 12px 32px rgba(15,23,42,0.07);
}

.image-card-title {

    padding:
        5px 8px 13px;

    color: #1e293b;

    font-size: 16px;

    font-weight: 850;
}

[data-testid="stImage"] {

    border-radius: 16px;
}


/* ============================================================
   DOWNLOAD BUTTONS
   ============================================================ */

.stDownloadButton > button {

    min-height: 50px !important;

    border-radius: 15px !important;

    font-weight: 800 !important;

    border:
        1px solid #dbeafe !important;

    box-shadow:
        0 8px 20px rgba(15,23,42,0.07) !important;

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease !important;
}

.stDownloadButton > button:hover {

    transform:
        translateY(-3px);

    box-shadow:
        0 14px 28px rgba(79,70,229,0.15) !important;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {

    margin-top: 50px;

    padding: 27px;

    text-align: center;

    border-radius: 24px;

    background:
        linear-gradient(
            135deg,
            #eef2ff,
            #fdf2f8
        );

    border:
        1px solid #e2e8f0;

    color: #475569;

    box-shadow:
        0 10px 28px rgba(15,23,42,0.05);
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 800px) {

    .hero {

        padding:
            40px 18px;

        border-radius:
            25px;
    }

    .hero-title {

        font-size:
            33px;
    }

    .badge {

        margin-top:
            8px;
    }

    .result-banner {

        flex-direction:
            column;

        align-items:
            flex-start;
    }
}

</style>
""")


# ============================================================
# HERO SECTION
# ============================================================

st.html("""
<div class="hero">

    <div class="hero-icon">
        🐾
    </div>

    <div class="hero-title">
        AI Image Segmentation Studio
    </div>

    <div class="hero-subtitle">
        TensorFlow + U-Net
    </div>

    <div class="hero-description">
        Upload an image and generate an AI-powered
        pixel-level segmentation mask with deep learning.
    </div>

    <div>

        <span class="badge">
            🧠 Deep Learning
        </span>

        <span class="badge">
            ⚡ TensorFlow
        </span>

        <span class="badge">
            🔬 U-Net
        </span>

        <span class="badge">
            🎯 Pixel Segmentation
        </span>

    </div>

</div>
""")


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = "best_oxford_pet_unet.keras"


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_segmentation_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    # --------------------------------------------------------
    # METHOD 1: Normal Keras loading
    # --------------------------------------------------------

    try:

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        return model, "standard"

    except Exception as first_error:

        # ----------------------------------------------------
        # METHOD 2: Legacy H5 compatibility repair
        # ----------------------------------------------------

        try:

            with h5py.File(MODEL_PATH, "r") as f:

                if "model_config" not in f.attrs:

                    raise RuntimeError(
                        "model_config not found inside model file."
                    )

                config = f.attrs["model_config"]

                if isinstance(config, bytes):

                    config = config.decode("utf-8")

                config = json.loads(config)

            # ------------------------------------------------
            # Remove legacy arguments
            # ------------------------------------------------

            def clean_config(obj):

                if isinstance(obj, dict):

                    if obj.get("class_name") == "DepthwiseConv2D":

                        obj["config"].pop(
                            "groups",
                            None
                        )

                    for value in obj.values():

                        clean_config(value)

                elif isinstance(obj, list):

                    for item in obj:

                        clean_config(item)

            clean_config(config)

            repaired_json = json.dumps(config)

            model = tf.keras.models.model_from_json(
                repaired_json
            )

            model.load_weights(MODEL_PATH)

            return model, "legacy-repaired"

        except Exception as second_error:

            raise RuntimeError(

                "Model could not be loaded.\n\n"

                f"Standard loading error:\n"
                f"{first_error}\n\n"

                f"Compatibility loading error:\n"
                f"{second_error}"
            )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model, load_method = load_segmentation_model()

    st.success(
        "✅ U-Net model loaded successfully!"
    )

except Exception as e:

    st.error(
        "❌ Model could not be loaded."
    )

    st.warning(
        "The model file is present, but TensorFlow/Keras "
        "compatibility needs attention."
    )

    with st.expander(
        "🔍 Technical error details"
    ):

        st.code(
            str(e),
            language="text"
        )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">'
        '🎛️ Segmentation Controls'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    threshold = st.slider(

        "🎯 Mask Threshold",

        min_value=0.10,

        max_value=0.90,

        value=0.50,

        step=0.05
    )

    st.markdown("---")

    st.markdown(
        "### 🧠 Model Information"
    )

    st.html(f"""
    <div class="sidebar-card">

        <b>Architecture</b><br>
        U-Net

        <br><br>

        <b>Framework</b><br>
        TensorFlow

        <br><br>

        <b>Input</b><br>
        128 × 128 × 3

        <br><br>

        <b>Output</b><br>
        128 × 128 × 1

        <br><br>

        <b>Activation</b><br>
        Sigmoid

        <br><br>

        <b>Loading</b><br>
        {load_method}

    </div>
    """)

    st.markdown("---")

    uploaded_file = st.file_uploader(

        "📤 Upload Image",

        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],

        help="Upload an image for segmentation."
    )


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    original = np.array(image)

    resized = image.resize(
        (128, 128),
        Image.Resampling.BILINEAR
    )

    img_array = np.array(
        resized
    ).astype(
        np.float32
    ) / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return original, img_array


# ============================================================
# PREDICTION
# ============================================================

def predict_mask(
    image,
    threshold_value
):

    original, input_image = preprocess_image(
        image
    )

    prediction = model.predict(
        input_image,
        verbose=0
    )

    prediction = np.squeeze(
        prediction
    )

    original_height, original_width = (
        original.shape[:2]
    )

    probability_image = Image.fromarray(

        np.clip(
            prediction * 255,
            0,
            255
        ).astype(
            np.uint8
        )
    )

    probability_image = probability_image.resize(

        (
            original_width,
            original_height
        ),

        Image.Resampling.BILINEAR
    )

    probability = np.array(
        probability_image
    ).astype(
        np.float32
    ) / 255.0

    binary_mask = (
        probability >= threshold_value
    ).astype(
        np.uint8
    ) * 255

    return (
        original,
        probability,
        binary_mask
    )


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(
    original,
    binary_mask
):

    overlay = np.zeros_like(
        original
    )

    overlay[:, :, 0] = 180
    overlay[:, :, 1] = 60
    overlay[:, :, 2] = 220

    mask_bool = binary_mask > 0

    result = original.copy()

    result[mask_bool] = (
        0.55 *
        original[mask_bool]

        +

        0.45 *
        overlay[mask_bool]
    ).astype(
        np.uint8
    )

    return result


# ============================================================
# DOWNLOAD IMAGE HELPER
# ============================================================

def image_to_bytes(array):

    image = Image.fromarray(
        array.astype(
            np.uint8
        )
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()


# ============================================================
# MAIN CONTENT
# ============================================================

if uploaded_file is None:

    st.html("""
    <div class="info-card">

        <div style="
            font-size:65px;
            margin-bottom:10px;
        ">
            📤
        </div>

        <h2 style="
            color:#172033;
            font-weight:900;
        ">
            Ready for Segmentation
        </h2>

        <p style="
            color:#64748b;
            font-size:17px;
        ">
            Upload an image from the sidebar to generate
            an AI-powered segmentation mask.
        </p>

        <div style="
            display:inline-block;
            padding:10px 18px;
            border-radius:999px;
            background:#f1f5f9;
            color:#64748b;
            font-size:13px;
            font-weight:700;
        ">
            JPG • JPEG • PNG • WEBP
        </div>

    </div>
    """)

else:

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error(
            "❌ Unable to read the uploaded image."
        )

        st.stop()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with st.spinner(
        "🧠 AI is analyzing the image..."
    ):

        original, probability, mask = (
            predict_mask(
                image,
                threshold
            )
        )

        overlay = create_overlay(
            original,
            mask
        )

    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    segmented_pixels = np.sum(
        mask > 0
    )

    total_pixels = mask.size

    segmentation_percentage = (
        segmented_pixels /
        total_pixels
    ) * 100

    average_confidence = (
        np.mean(probability) *
        100
    )


    # ========================================================
    # RESULT HEADER
    # ========================================================

    st.html(f"""
    <div class="result-banner">

        <div>

            <div class="result-title">
                ✨ AI Segmentation Complete
            </div>

            <div class="result-meta">
                Your image has been analyzed using
                the trained U-Net model.
            </div>

        </div>

        <div class="result-status">
            ● ANALYSIS READY
        </div>

    </div>
    """)


    # ========================================================
    # ANALYTICS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Segmentation Analytics'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Key statistics generated from the predicted mask.'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.html(f"""
        <div class="metric-card">

            <div class="metric-icon">
                🎯
            </div>

            <div class="metric-value">
                {segmentation_percentage:.1f}%
            </div>

            <div class="metric-label">
                Segmented Area
            </div>

        </div>
        """)


    with col2:

        st.html(f"""
        <div class="metric-card">

            <div class="metric-icon">
                🧠
            </div>

            <div class="metric-value">
                {average_confidence:.1f}%
            </div>

            <div class="metric-label">
                Mean Prediction
            </div>

        </div>
        """)


    with col3:

        st.html("""
        <div class="metric-card">

            <div class="metric-icon">
                🖼️
            </div>

            <div class="metric-value">
                128×128
            </div>

            <div class="metric-label">
                Model Input
            </div>

        </div>
        """)


    with col4:

        st.html(f"""
        <div class="metric-card">

            <div class="metric-icon">
                ⚙️
            </div>

            <div class="metric-value">
                {threshold:.2f}
            </div>

            <div class="metric-label">
                Threshold
            </div>

        </div>
        """)


    # ========================================================
    # EXTRA STATISTICS
    # ========================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    info1, info2 = st.columns(2)


    with info1:

        st.html(f"""
        <div class="confidence-card">

            <div class="confidence-label">
                SEGMENTED PIXELS
            </div>

            <div class="confidence-number">
                {segmented_pixels:,}
            </div>

            <div class="confidence-label">
                detected pixels out of
                {total_pixels:,} total pixels
            </div>

        </div>
        """)


    with info2:

        st.html(f"""
        <div class="confidence-card">

            <div class="confidence-label">
                AVERAGE MODEL SCORE
            </div>

            <div class="confidence-number">
                {average_confidence:.1f}%
            </div>

            <div class="confidence-label">
                average probability across
                the complete prediction map
            </div>

        </div>
        """)


    # ========================================================
    # VISUAL COMPARISON
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🖼️ Visual Comparison'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Compare the original image, segmentation mask, '
        'and AI-generated overlay.'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.html("""
        <div class="image-card">

            <div class="image-card-title">
                📷 Original Image
            </div>

        </div>
        """)

        st.image(
            original,
            use_container_width=True
        )


    with col2:

        st.html("""
        <div class="image-card">

            <div class="image-card-title">
                🎭 Segmentation Mask
            </div>

        </div>
        """)

        st.image(
            mask,
            use_container_width=True
        )


    with col3:

        st.html("""
        <div class="image-card">

            <div class="image-card-title">
                ✨ AI Overlay
            </div>

        </div>
        """)

        st.image(
            overlay,
            use_container_width=True
        )


    # ========================================================
    # PROBABILITY MAP
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🔥 Prediction Confidence Map'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Brighter regions represent higher model probability.'
        '</div>',
        unsafe_allow_html=True
    )


    probability_display = (
        np.clip(
            probability * 255,
            0,
            255
        ).astype(
            np.uint8
        )
    )


    prob_col1, prob_col2 = st.columns(
        [2.8, 1]
    )


    with prob_col1:

        st.html("""
        <div class="image-card">

            <div class="image-card-title">
                🔥 Probability Heatmap
            </div>

        </div>
        """)

        st.image(
            probability_display,
            caption="Model probability map",
            use_container_width=True
        )


    with prob_col2:

        st.html(f"""
        <div class="confidence-card">

            <div class="confidence-label">
                MODEL SUMMARY
            </div>

            <br>

            <div class="mini-info">
                <b>Architecture</b><br>
                U-Net
            </div>

            <br>

            <div class="mini-info">
                <b>Framework</b><br>
                TensorFlow / Keras
            </div>

            <br>

            <div class="mini-info">
                <b>Threshold</b><br>
                {threshold:.2f}
            </div>

            <br>

            <div class="mini-info">
                <b>Output</b><br>
                128 × 128 × 1
            </div>

        </div>
        """)


    # ========================================================
    # DOWNLOAD SECTION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📥 Export Your Results'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Save your generated segmentation results as PNG.'
        '</div>',
        unsafe_allow_html=True
    )


    download_col1, download_col2 = st.columns(2)


    with download_col1:

        st.download_button(

            label=
            "⬇️ Download Segmentation Mask",

            data=
            image_to_bytes(mask),

            file_name=
            "segmentation_mask.png",

            mime=
            "image/png",

            use_container_width=True
        )


    with download_col2:

        st.download_button(

            label=
            "⬇️ Download AI Overlay",

            data=
            image_to_bytes(overlay),

            file_name=
            "segmentation_overlay.png",

            mime=
            "image/png",

            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">

    <div style="
        font-size:18px;
        font-weight:900;
    ">
        🐾 AI Image Segmentation Studio
    </div>

    <div style="
        font-size:13px;
        margin-top:7px;
        color:#64748b;
    ">
        Smart pixel-level analysis powered by Deep Learning
    </div>

    <br>

    <span>
        Built with ❤️ using
        <b>TensorFlow + U-Net + Streamlit</b>
    </span>

    <br><br>

    <span style="
        font-size:12px;
        color:#94a3b8;
    ">
        Pixel-Level Image Segmentation
    </span>

</div>
""")