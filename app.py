import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

@st.cache_resource
def load_my_model():
    return load_model('cat_detector_final.keras')

model = load_my_model()

st.title("🐱 Cat or Not? 🐶")
st.write("Upload an image and I'll tell you if it's a cat!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Your Image', use_column_width=True)

    img = image.resize((224, 224))
    arr = img_to_array(img)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    with st.spinner('Thinking...'):
        pred = model.predict(arr, verbose=0)[0][0]

    if pred > 0.5:
        st.error(f"❌ No, it's not a cat! 🐶 (Confidence: {pred*100:.2f}%)")
    else:
        st.success(f"✅ Yes, it's a cat! 🐱 (Confidence: {(1-pred)*100:.2f}%)")