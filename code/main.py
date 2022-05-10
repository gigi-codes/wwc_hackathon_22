import streamlit as st
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow import keras
    
st.title('Percentage probability of Glaucoma')

page = st.sidebar.selectbox(
    'Select a page:',
    ('About', 'Make a prediction')
)

if page == 'About':
    st.write('here is my model')
    st.write('get in touch with me at:')

if page == 'Make a prediction':
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = np.asarray(Image.open(uploaded_file).resize((178,178)))
        image_data = image_data.reshape(1, 178, 178, 3)
        st.image(image_data)
        model = keras.models.load_model('../models/model_01/')
        prediction = model.predict(image_data)[0][0]
        st.write(f'The probability that this retina is pathological for Glaucoma is {prediction*100}%')