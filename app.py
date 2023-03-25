import numpy as np
import streamlit as st
from skimage.io import imread
from skimage.transform import resize
import pickle
import webbrowser as wb

model1 = pickle.load(open('Models/KCRaa.pkl', 'rb'))  # LogisticRegression
model2 = pickle.load(open('Models/KCRkaa.pkl', 'rb'))
model3 = pickle.load(open('Models/KCRnum.pkl', 'rb'))

characters_list_1 = ['ಅ', 'ಆ', 'ಇ', 'ಈ', 'ಉ',
                     'ಊ', 'ಋ', 'ಎ', 'ಏ', 'ಐ', 'ಒ', 'ಓ', 'ಔ', 'ಅಂ', 'ಅಃ']
characters_list_2 = {'17': 'ಕ್', '18': 'ಕ', '19': 'ಕಾ', '20': 'ಕಿ',
                     '21': 'ಕೀ', '22': 'ಕು', '23': 'ಕೂ', '24': 'ಕೃ', '25': 'ಕೈ', '26': 'ಕೆ', '27': 'ಕೇ', '28': 'ಕೈ', '29': 'ಕೊ', '30': 'ಕೋ', '31': 'ಕೌ', '32': 'ಕಂ', '33': 'ಕಃ'}
characters_list_3 = {'648': '೦', '649': '೧', '650': '೨',
                     '651': '೩', '652': '೪', '653': '೫', '654': '೬', '655': '೭', '656': '೮', '657': '೯'}


def tranform_image(img):
    feature = []
    tranform_img = resize(img, (150, 150, 3))
    flatten_img = tranform_img.flatten()
    feature.append(flatten_img)
    return np.array(feature)


# --------------UI Beginning------------
# SideBar

side_bar = st.sidebar.radio(
    'Select Characters', ['Characters(ಅ-ಅಃ)', 'Characters(ಕ-ಕಃ)', 'Characters(೦-೯)', 'About'])


# page 1
try:
    if side_bar == 'Characters(ಅ-ಅಃ)':
        st.title('Kannada Character Recoginition (ಅ-ಅಃ)')
        st.header('Pick a Image (from test_img_aa)')
        img_name = st.file_uploader('Select Image')
        try:
            st.image(f'test_img_aa/{img_name.name}')
            img_ = imread(f'test_img_aa/{img_name.name}')
        except:
            st.image(f'test_img/test_img/{img_name.name}')
            img_ = imread(f'test_img/test_img/{img_name.name}')
        # Predict Button
        if st.button('Predict Character'):
            try:
                # LR
                res1 = (model1.predict(tranform_image(img_)))
                st.header(
                    f'Predicted Character is {characters_list_1[res1[0]-1]}')
            except:
                st.info('Please try again.')
except:
    pass

# page 2
if side_bar == 'Characters(ಕ-ಕಃ)':
    st.title('Kannada Character Recoginition (ಕ-ಕಃ)')
    st.header('Pick a Image (from test_img_ka)')
    img_name = st.file_uploader('Select Image')
    try:
        st.image(f'test_img_ka/{img_name.name}')
        img_ = imread(f'test_img_ka/{img_name.name}')
    except:
        st.image(f'test_img/test_img/{img_name.name}')
        img_ = imread(f'test_img/test_img/{img_name.name}')
    # Predict Button
    if st.button('Predict Character'):
        try:
            # LR
            res2 = (model2.predict(tranform_image(img_)))[0]
            res2 = str(res2)
            st.header(
                f'Character is {characters_list_2[res2]}')
        except:
            st.info('Please try again.')


# page 3
try:
    if side_bar == 'Characters(೦-೯)':
        st.title('Kannada Character Recoginition (೦-೯)')
        st.title('Pick a Image (from test_img_num)')
        img_name = st.file_uploader('Select Image')
        try:
            st.image(f'test_img_num/{img_name.name}')
            img_ = imread(f'test_img_num/{img_name.name}')
        except:
            st.image(f'test_img/test_img/{img_name.name}')
            img_ = imread(f'test_img/test_img/{img_name.name}')
        # Predict Button
        if st.button('Predict Character'):
            try:
                # LR
                res3 = (model3.predict(tranform_image(img_)))[0]
                res3 = str(res3)
                st.header(
                    f'Predicted Character is {characters_list_3[res3]}')
            except:
                st.info('Please try again.')
except:
    pass

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
