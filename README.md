# 📩 Spam Message Classifier (NLP + Machine Learning)

## Overview

This project is a Machine Learning based Spam Message Classifier built using Natural Language Processing (NLP) techniques.  

The model classifies text messages into two categories:

- Ham (Not Spam)  
- Spam  

The project covers the complete workflow from data preprocessing and exploratory data analysis to model training and deployment using Streamlit.

---

## Problem Statement

Spam messages are common in email and SMS platforms. The objective of this project is to build a reliable model that can automatically detect whether a message is spam or not based on its text content.

---

## Technologies Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- NLTK  
- Scikit-learn  
- Streamlit  
- Pickle  

---

## Project Workflow

### 1. Data Cleaning
- Removed unnecessary columns  
- Renamed columns for clarity  
- Removed duplicate records  
- Converted categorical labels (ham/spam) into numeric format  

### 2. Exploratory Data Analysis (EDA)
- Checked class distribution  
- Analyzed number of characters, words, and sentences  
- Visualized distributions using charts  

### 3. Text Preprocessing
- Converted text to lowercase  
- Tokenization  
- Removed stopwords  
- Removed punctuation  
- Applied stemming  

### 4. Feature Engineering
- Used TF-IDF Vectorizer  
- Applied unigram and bigram features  
- Tuned parameters like `min_df`, `max_df`, and `ngram_range`

### 5. Model Training
Compared different Naive Bayes models:
- Gaussian Naive Bayes  
- Multinomial Naive Bayes  
- Bernoulli Naive Bayes  

Multinomial Naive Bayes performed best for this text classification task.

### 6. Model Evaluation
Evaluated performance using:
- Accuracy  
- Confusion Matrix  
- Precision  
- Recall  
- F1 Score  

### 7. Deployment
- Saved trained model and vectorizer using pickle  
- Built an interactive web application using Streamlit  

---
Spam_Classifier/
│
├── spam.csv
├── model.pkl
├── vectorizer.pkl
├── app.py
├── training_notebook.ipynb
└── README.md

---

## How to Run the Project

### 1. Install Dependencies

### 2. Run the Application

---

## Example

**Input:**  
Congratulations! You have won a free prize.

**Output:**  
Spam

---

## Future Improvements

- Add Logistic Regression and SVM for comparison  
- Implement cross-validation  
- Improve UI design  
- Deploy on a cloud platform  
- Experiment with deep learning models  

---

## Author

Anushka Vishwakarma
BCA Student  
Lucknow, India

## Project Structure

