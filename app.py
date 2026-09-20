import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path='cat_detector.tflite')
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("🐱 Cat or Not? 🐶")
st.write("Upload an image and I'll tell you if it's a cat!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Your Image', use_column_width=True)

    # ---- Preprocessing: EXACTLY like training ----
    img = image.resize((224, 224))
    arr = img_to_array(img)                    # same as training
    arr = preprocess_input(arr)                # same as training
    arr = np.expand_dims(arr, axis=0)          # same as training
    arr = arr.astype(np.float32)               # TFLite needs float32

    # ---- Predict ----
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # ---- 0 = cat, 1 = dog (matches your training) ----
    if pred > 0.5:
        st.error(f"❌ No, it's not a cat! 🐶 (Confidence: {pred*100:.2f}%)")
    else:
        st.success(f"✅ Yes, it's a cat! 🐱 (Confidence: {(1-pred)*100:.2f}%)")
