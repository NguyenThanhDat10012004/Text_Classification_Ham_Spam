# Pine_line
![image](https://github.com/user-attachments/assets/abf36381-29cb-4995-95d3-61736ce3f460)
# Cài Đặt các thư viện cần thiết: 
```python
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
```
# Xử lí data
## chia data thành 2 mục là label và data
```python
# tách riêng phần data, label để dễ xử lí
DATASET_PATH = '/content/2cls_spam_text_cls.csv'
df = pd.read_csv(DATASET_PATH)
messages = df['Message'].values.tolist()
labels = df['Category'].values.tolist()
```
## processsing data
```python
# xử lí nhãn về nhị phân
le = LabelEncoder()
y = le.fit_transform(labels)
print(f'Classes: {le.classes_}')
print(f'Encoded labels: {y}')
```
```python
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
```
# Chuyển đổi các đoạn text sang vector số
```python
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
```
# Chia tập dataset theo tỉ lệ 7/2/1
```python
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
```
# Train model
```python
%%time
model = GaussianNB()
print('Start training...')
model = model.fit(X_train, y_train)
print('Training completed!')
```
# Validation model
```python
# đánh giá mô hình dựa trên tập validation và tập test
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Val accuracy: {val_accuracy}')
print(f'Test accuracy: {test_accuracy}')
```
# Predict one example
```python
test_input = 'I am actually thinking a way of doing something useful'
prediction_cls = predict(test_input, model, dictionary)
print(f'Prediction: {prediction_cls}')
```
### Note 1: sử dụng Bayes
![image](https://github.com/user-attachments/assets/00ecbe03-297c-467d-bb9a-2e74afc87247)
### Note 2: Ngoài sử dụng bayes để làm ta cũng có thể sử dụng tf-idf để nâng cao độ chính xác cho model bằng cách kết hợp tf-idf với mô hình linear regression hoặc mô hình svm


