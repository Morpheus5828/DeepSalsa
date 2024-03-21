import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from tensorflow.keras.preprocessing.image import img_to_array, load_img


@st.cache(allow_output_mutation=True)
def load():
    model_path = "dps_model.h5"
    options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    model = tf.keras.models.load_model(model_path, compile=False, options=options)
    return model

model = load()

def predict(upload):

    img = Image.open(upload)
    img = np.asarray(img)
    img = img[:, :, :3]
    img_resize = cv2.resize(img, (657, 413))
    img_resize = np.expand_dims(img_resize, axis=0)
    pred = model.predict(img_resize)

    rec = pred[0][0]

    return rec

st.title("DeepSalsaApp")

upload = st.file_uploader("Upload image",
                           type=['png', 'jpeg', 'jpg'])

c1, c2 = st.columns(2)

if upload:
    rec = predict(upload)
    prob_recyclable = rec * 100
    prob_organic = (1-rec)*100

    c1.image(Image.open(upload))
    if prob_recyclable > 50:
        c2.write(f"I am sure {prob_recyclable:.2f} % picture contains sea")
    else:
        c2.write(f"I am sure {prob_organic:.2f} % picture doesn't contains sea")
