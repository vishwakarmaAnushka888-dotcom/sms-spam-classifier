import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
def transform_text(text):
    # Lowercase
    text = text.lower()
    text=nltk.word_tokenize(text)
    top_words = set(stopwords.words('english'))
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in top_words and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)
st.title('Email Spam Classifier')
sms=st.text_input('Enter the  message')
if st.button('Predict'):
    transform_sms=transform_text(sms)
    vector_input=tfidf.transform([transform_sms])
    result=model.predict(vector_input)[0]
    if result==0:
        st.header('Not Spam')
    elif result==1:
        st.header('Spam')