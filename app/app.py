import streamlit as st
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Fish Image Classification",
    page_icon="🐟",
    layout="centered"
)

# =========================
# LOAD MODEL
# =========================

model = load_model("best_model.h5")

# =========================
# CLASS LABELS
# =========================

class_names = [
    'animal fish',
    'animal fish bass',
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]

# =========================
# TITLE
# =========================

st.title("🐟 Multiclass Fish Image Classification")

st.markdown("""
Upload a fish image and the model will predict the fish category using Deep Learning.
""")

# =========================
# IMAGE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload Fish Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION
# =========================

if uploaded_file is not None:

    # Display uploaded image
    img = Image.open(uploaded_file)

    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize image
    img = img.resize((224, 224))

    # Convert to array
    img_array = image.img_to_array(img)

    # Normalize
    img_array = img_array / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions)

    predicted_class = class_names[predicted_index]

    confidence_score = np.max(predictions) * 100

    # =========================
    # OUTPUT
    # =========================

    st.success(f"Predicted Fish Category: {predicted_class}")

    st.info(f"Confidence Score: {confidence_score:.2f}%")

    # =========================
    # PROBABILITY TABLE
    # =========================

    st.subheader("Prediction Probabilities")

    for i, class_name in enumerate(class_names):

        probability = predictions[0][i] * 100

        st.write(f"{class_name}: {probability:.2f}%")

        st.progress(float(predictions[0][i]))

# =========================
# FOOTER
# =========================

st.markdown("---")

st.markdown(
    "Built using TensorFlow, Keras, and Streamlit"
)