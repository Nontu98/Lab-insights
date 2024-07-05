import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import os

# Absolute path to the directory containing the files
directory = 'c:/Users/Nontu/Desktop/CLassification_project-Team-EG5/DATA'

# Load the trained model
model_path = os.path.join(directory, 'svm_model.pkl')
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Load the TfidfVectorizer
vectorizer_path = os.path.join(directory, 'tfidf_vectorizer.pkl')
with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit app
st.title('News Article Classification')
st.write('Enter a news article and the model will predict its category.')

# Input news article
article = st.text_area('News Article', '')

# Predict the category
if st.button('Classify'):
    if article:
        article_transformed = vectorizer.transform([article])
        prediction = model.predict(article_transformed)[0]
        st.write(f'The predicted category is: **{prediction}**')
    else:
        st.write('Please enter a news article.')

# To run the app: streamlit run app.py

