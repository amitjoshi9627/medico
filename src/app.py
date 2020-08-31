import streamlit as st
from plant_disease_classifier import get_prediction
from PIL import Image

st.title("Plant Pathology")
uploaded_file = st.file_uploader("Choose an Image")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, width=300)
    result = get_prediction(uploaded_file)
    if st.button("Check Result"):
        st.write("## __{}__".format(result))
