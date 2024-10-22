# -*- coding: utf-8 -*-
"""Untitled56.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1liluV-uVCtHe0ddMWQFqwtt6_ZGpjTeq
"""

!gdown --id 1N7rk-kfnDFIGMeX0ROVTjKh71gcgx-7R

#import các thứ viện cần thiết, trong đó có pandas xử lí file csv, nltk xử lý phần nlp, sklearn hỗ trợ cung cấp các model.
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# tách riêng phần data, label để dễ xử lí
DATASET_PATH = '/content/2cls_spam_text_cls.csv'
df = pd.read_csv(DATASET_PATH)
messages = df['Message'].values.tolist()
labels = df['Category'].values.tolist()

# xử lí nhãn về nhị phân
le = LabelEncoder()
y = le.fit_transform(labels)
print(f'Classes: {le.classes_}')
print(f'Encoded labels: {y}')

# processing data bằng cách chuyển hết về thường, loại bỏ dấu, tokenize, xoá bỏ từ dừng không có ý nghĩa
def lowercase(text):
    return text.lower()

def punctuation_removal(text):
    translator = str.maketrans('', '', string.punctuation)

    return text.translate(translator)

def tokenize(text):
    return nltk.word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = nltk.corpus.stopwords.words('english')

    return [token for token in tokens if token not in stop_words]

def stemming(tokens):
    stemmer = nltk.PorterStemmer()

    return [stemmer.stem(token) for token in tokens]

def preprocess_text(text):
    text = lowercase(text)
    text = punctuation_removal(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)

    return tokens

messages = [preprocess_text(message) for message in messages]

# tạo một từ điển chứa các từ không trùng lặp, sau đó tạo vector cho mỗi tokens đó.
def create_dictionary(messages):
    dictionary = []

    for tokens in messages:
        for token in tokens:
            if token not in dictionary:
                dictionary.append(token)

    return dictionary

def create_features(tokens, dictionary):
    features = np.zeros(len(dictionary))

    for token in tokens:
        if token in dictionary:
            features[dictionary.index(token)] += 1

    return features

dictionary = create_dictionary(messages)
X = np.array([create_features(tokens, dictionary) for tokens in messages])

# chia tập dữ liệu 7/2/1
VAL_SIZE = 0.2
TEST_SIZE = 0.125
SEED = 0

X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=VAL_SIZE,
                                                  shuffle=True,
                                                  random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                    test_size=TEST_SIZE,
                                                    shuffle=True,
                                                    random_state=SEED)

# Commented out IPython magic to ensure Python compatibility.
# # train model
# %%time
# model = GaussianNB()
# print('Start training...')
# model = model.fit(X_train, y_train)
# print('Training completed!')

# đánh giá mô hình dựa trên tập validation và tập test
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Val accuracy: {val_accuracy}')
print(f'Test accuracy: {test_accuracy}')

# hàm dự đoán
def predict(text, model, dictionary):
    processed_text = preprocess_text(text)
    features = create_features(text, dictionary)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    prediction_cls = le.inverse_transform(prediction)[0]

    return prediction_cls

test_input = 'I am actually thinking a way of doing something useful'
prediction_cls = predict(test_input, model, dictionary)
print(f'Prediction: {prediction_cls}')

"""**Cách số 2 cải thiện độ chính xác, sử dụng thuật toán tf-idf**"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Tách đặc trưng và nhãn
X = df['Message']  # Cột chứa văn bản
y = df['Category'].map({'ham': 0, 'spam': 1})  # Chuyển nhãn thành số (ham = 0, spam = 1)

# Chuyển văn bản thành đặc trưng TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95)
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Chia tập dữ liệu thành train, val, test
X_train, X_temp, y_train, y_temp = train_test_split(X_tfidf, y, test_size=0.3, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

# Chuyển đổi dữ liệu từ dạng sparse matrix thành array để GaussianNB có thể xử lý
X_train_array = X_train.toarray()
X_val_array = X_val.toarray()
X_test_array = X_test.toarray()

# Khởi tạo và huấn luyện mô hình Gaussian Naive Bayes
model = GaussianNB()
model.fit(X_train_array, y_train)

# Đánh giá mô hình trên tập validation
y_val_pred = model.predict(X_val_array)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy}')

# Đánh giá mô hình trên tập test
y_test_pred = model.predict(X_test_array)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy}')

import joblib

# Lưu mô hình và TF-IDF
joblib.dump(model, 'spam_classifier_nb.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

"""**Phương pháp khác**"""

tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.95)
X_tfidf = tfidf_vectorizer.fit_transform(X)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
X_tfidf = tfidf_vectorizer.fit_transform(X)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train_array, y_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Val accuracy: {val_accuracy}')
print(f'Test accuracy: {test_accuracy}')

from sklearn.svm import SVC

model = SVC()
model.fit(X_train_array, y_train)
# Chuyển đổi tập validation và test từ dạng sparse matrix thành dense array
X_val_array = X_val.toarray()
X_test_array = X_test.toarray()

# Dự đoán trên tập validation và test sau khi chuyển đổi thành array
y_val_pred = model.predict(X_val_array)
y_test_pred = model.predict(X_test_array)

# Tính toán độ chính xác
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Validation Accuracy: {val_accuracy}')
print(f'Test Accuracy: {test_accuracy}')

test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy}')

