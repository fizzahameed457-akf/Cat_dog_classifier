import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input, MobileNetV2, decode_predictions
)

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path='cat_detector.tflite')
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def load_imagenet_model():
    return MobileNetV2(weights='imagenet')

interpreter = load_tflite_model()
imagenet_model = load_imagenet_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Keywords from ImageNet's 1000 classes that mean "cat" or "dog"
PET_KEYWORDS = [
    'cat', 'kitten', 'tabby', 'tiger_cat', 'Persian', 'Siamese', 'Egyptian',
    'lynx', 'cougar', 'puma', 'panther',
    'dog', 'puppy', 'retriever', 'terrier', 'spaniel', 'hound', 'shepherd',
    'bulldog', 'poodle', 'chihuahua', 'malamute', 'husky', 'corgi', 'dalmatian',
    'beagle', 'collie', 'pug', 'boxer', 'rottweiler', 'doberman', 'pinscher',
    'mastiff', 'coyote', 'wolf', 'fox', 'Shiba', 'Akita', 'Schnauzer',
    'Pekinese', 'Maltese', 'Papillon', 'Pomeranian', 'Samoyed', 'Newfoundland',
    'Weimaraner', 'Vizsla', 'Pointer', 'Setter', 'Briard', 'Kelpie',
    'Komondor', 'Kuvasz', 'Leonberg', 'Lhasa', 'Malinois', 'Saluki',
    'whippet', 'dingo'
]

def imagenet_check(image):
    """Returns (is_pet, top_label)"""
    img = image.resize((224, 224))
    arr = preprocess_input(img_to_array(img))
    arr = np.expand_dims(arr, axis=0)
    preds = imagenet_model.predict(arr, verbose=0)
    top5 = decode_predictions(preds, top=5)[0]
    for _, label, _ in top5:
        label_lower = label.lower()
        for kw in PET_KEYWORDS:
            if kw.lower() in label_lower:
                return True, label
    return False, top5[0][1]

st.title("🐱 Cat or Dog 🐶")
st.write("Upload an image and I'll tell you if it's a cat or a dog!")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "heic", "heif", "webp", "bmp", "jfif", "tif", "tiff", "gif"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Your Image', use_container_width=True)

    with st.spinner("Analyzing..."):
        is_pet, label = imagenet_check(image)

    if not is_pet:
        pretty = label.replace('_', ' ')
        st.warning(f"Neither a cat nor a dog! (I think it's a **{pretty}**)")
    else:
        # Gatekeeper passed — run YOUR custom model
        img = image.resize((224, 224))
        arr = preprocess_input(img_to_array(img))
        arr = np.expand_dims(arr, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

        if pred > 0.5:
            st.error(f"Oh, it's a cat! 🐶 (Confidence: {pred*100:.2f}%)")
        else:
            st.success(f"Oh, it's a cat! 🐱 (Confidence: {(1-pred)*100:.2f}%)")
