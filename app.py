import numpy as np
import streamlit as st
from skimage.io import imread, imshow
from skimage.transform import resize
import pickle
from streamlit_drawable_canvas import st_canvas
import webbrowser as wb
import os

model1 = pickle.load(open('Models/KCR(LR).pkl', 'rb'))  # LogisticRegression
model2 = pickle.load(open('Models/KCR(RF).pkl', 'rb'))  # RandomForest

characters_list = ['ಅ', 'ಆ', 'ಇ', 'ಈ', 'ಉ',
                   'ಊ', 'ಋ', 'ಎ', 'ಏ', 'ಐ', 'ಒ', 'ಓ', 'ಔ', 'ಅಂ', 'ಅಃ']


def tranform_image(img):
    feature = []
    tranform_img = resize(img, (150, 150, 3))
    flatten_img = tranform_img.flatten()
    feature.append(flatten_img)
    return np.array(feature)


# --------------UI Beginning------------
st.title('Kannada Character Recoginition')

# SideBar

side_bar = st.sidebar.radio(
    'Select Input Mode', ['Image Picker', 'Canvas(Beta)', 'About'])


# Image Picker

try:
    if side_bar == 'Image Picker':
        st.title('Pick a Image')
        img_name = st.file_uploader('Select Image')
        try:
            st.image(f'test_img/{img_name.name}')
            img_ = imread(f'test_img/{img_name.name}')
        except:
            st.image(f'test_img/test_img/{img_name.name}')
            img_ = imread(f'test_img/test_img/{img_name.name}')
        # Predict Button
        if st.button('Predict Class'):
            try:
                # LR
                res1 = (model1.predict(tranform_image(img_)))
                st.header(f'Character is {characters_list[res1[0]-1]} by LR')
                # RF
                res2 = (model1.predict(tranform_image(img_)))
                st.header(f'Character is {characters_list[res2[0]-1]} by RF')

            except:
                st.info('Please try again.')
except:
    pass

if side_bar == 'Canvas(Beta)':
    #   Canvas
    with st.form('Canvas'):
        st.text('Note : Test Release Version')
        canvas_result = st_canvas(background_color='white',
                                  width=400, height=200, stroke_width=15)

        # Predict Button
        if st.form_submit_button('Predict Class'):
            try:
                # LR
                res1 = (model1.predict(tranform_image(canvas_result.image_data)))
                st.header(f'Character is {characters_list[res1[0]-1]} by LR')
                # RF
                res2 = (model1.predict(tranform_image(canvas_result.image_data)))
                st.header(f'Character is {characters_list[res2[0]-1]} by RF')

            except:
                st.info('Please try again.')

# About
if side_bar == 'About':
    with st.form('about'):
        st.title('About Project')

        st.text('''
        Kannada OCR (Optical Character Recognition) with ML (Machine Learning)
        classification algorithm involves training a machine learning model to recognize
        and classify Kannada characters from scanned or digital images.OCR technology
        has become increasingly important in recent years due to the growth of
        digitization and the need to process large volumes of documents quickly and
        accurately.OCR technology is used in a variety of applications, including data
        entry, document archiving, and information retrieval.Kannada OCR with ML
        classification algorithm is especially important for preserving and digitizing
        Kannada literature and documents.Kannada is a Dravidian language spoken
        predominantly in the Indian state of Karnataka.Kannada literature is rich and
        diverse, with a history that spans over a thousand years.However, much of this
        literature remains in print form and is not easily accessible to the wider
        public.OCR technology can help to digitize these documents, making them more
        easily accessible to scholars and researchers.ML classification algorithms are
        used to classify Kannada characters based on their visual features. These
        algorithms learn from a set of training data, and then use this knowledge to
        classify new data. Some popular ML classification algorithms for OCR include
        Support Vector Machines (SVM), Random Forests, and Convolutional Neural
        Networks (CNN).
                ''')

        if st.form_submit_button('Report Bug'):
            wb.open(
                "https://gmail.google.com/mail/?view=cm&fs=1&to=akash.hiremath25@gmail.com.com&su=Bug%20Report")
