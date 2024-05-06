import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load pre-trained model and data
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Load background image
background_image = Image.open('pic2.jpg')

# Display background image with Streamlit
st.image(background_image, use_column_width=True)

# Add fashion animations with HTML and CSS
st.markdown(
    """
    <style>
    @keyframes rotate {
        from {
            transform: rotate(0deg);
        }
        to {
            transform: rotate(360deg);
        }
    }

    .fashion-animation {
        width: 100px;
        height: 100px;
        background-color: #FA8072;
        position: absolute;
        top: 50%;
        left: 45%;
        animation: rotate 5s linear infinite;
    }
    </style>
    """
    , unsafe_allow_html=True
)

# Add fashion animations to the Streamlit app
st.markdown('<div class="fashion-animation"></div>', unsafe_allow_html=True)

# Change the title to "DRESSED RIGHT" with shadowed black font
st.markdown(
    """
    <style>
    .title {
        text-shadow: 2px 2px #000000;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title('Fashion Recommendation System')

# Define function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

# Define function for feature extraction
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Define function for recommendation
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# Steps: file upload -> save -> display
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption='Uploaded Image', use_column_width=True)

        # Extract features from uploaded image
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

        # Recommendation
        indices = recommend(features, feature_list)

        # Display recommended images in columns
        cols = st.columns(5)
        for i, col in enumerate(cols):
            col.image(filenames[indices[0][i]])

    else:
        st.error("Some error occurred in file upload")
