import numpy as np
import pandas as pd
import re
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from nltk.classify import NaiveBayesClassifier
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.layers import Dense
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
nltk.download('stopwords')
location = r'C:REPLACE WITH YOUR PATH OF EXCEL FILE.xlsx'
xls = pd.ExcelFile(location)
sheet_names = xls.sheet_names
print(sheet_names)
df = pd.read_excel(location)
# Convert to lowercase
df['review'] = df['review'].str.lower()
# Remove stop words
stop_words = set(stopwords.words("english"))
df['review'] = df['review'].apply(lambda x: ' '.join([word for word in x.split() 
if word not in stop_words]))
# Remove specific characters and patterns
df['review'] = df['review'].apply(lambda x: re.sub(r'[?;():0-9.<>!‘`~@]', '', x))
df['review'] = df['review'].apply(lambda x: re.sub(r'<br /><br />', '', x))
# Stemming
stemmer = PorterStemmer()
df['review'] = df['review'].apply(lambda x: ' '.join([stemmer.stem(word) for word
in x.split()]))
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['review'], 
df['sentiment'], test_size=0.15, random_state=23)
print('pre-processing is done !!')
#######
count_vectorizer = CountVectorizer(max_features=10000)
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)
nb_count = MultinomialNB()
nb_count.fit(X_train_count, y_train)
y_pred_count = nb_count.predict(X_test_count)
# Calculate precision, recall, and F1 score for CountVectorizer
report_count = classification_report(y_test, y_pred_count)
print("Bag-of-Words Report using Native Bayes:")
print(report_count)
################################
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = nb_tfidf.predict(X_test_tfidf)
report_tfidf = classification_report(y_test, y_pred_tfidf)
print("TF-IDF Report using Native Bayes:")
print(report_tfidf)
##############################################################################
######################### LOGISITIC REGRESSION ###############################
lr_count = LogisticRegression()
lr_count.fit(X_train_count, y_train)
## BAG OF WORDs ##
y_pred_count_lr = lr_count.predict(X_test_count)
report_count_lr = classification_report(y_test, y_pred_count_lr)
print("Bag-of-Words Report with Logistic Regression:")
print(report_count_lr) 
## TF-IDF ##
lr_tfidf = LogisticRegression()
lr_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf_lr = lr_tfidf.predict(X_test_tfidf)
report_tfidf_lr = classification_report(y_test, y_pred_tfidf_lr)
print("TF-IDF Report with Logistic Regression:")
print(report_tfidf_lr)
##############################################################################
############################ NEURAL NETWORK ##################################
num_classes = len(set(y_train))
model_count = Sequential()
model_count.add(Dense(64, activation='relu', input_dim=X_train_count.shape[1]))
model_count.add(Dropout(0.5))
model_count.add(Dense(64, activation='relu'))
model_count.add(Dropout(0.5))
model_count.add(Dense(num_classes, activation='softmax'))
model_count.compile(loss='categorical_crossentropy', optimizer='adam', 
metrics=['accuracy'])
label_names = ['positive', 'negitive']
label_encoder = LabelEncoder()
# Encode the labels
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
# Get the number of classes
num_classes = len(label_encoder.classes_)
y_train_onehot = tf.keras.utils.to_categorical(y_train_encoded)
y_test_onehot = tf.keras.utils.to_categorical(y_test_encoded)
model_count.fit(X_train_count.toarray(), y_train_onehot, epochs=10, 
batch_size=32, validation_data=(X_test_count.toarray(), y_test_onehot))
y_pred_count_nn = model_count.predict(X_test_count.toarray())
y_pred_count_nn = np.argmax(y_pred_count_nn, axis=1)
y_pred_count_nn = label_encoder.inverse_transform(y_pred_count_nn)
report_count_nn = classification_report(y_test, y_pred_count_nn)
# Neural Network with TF-IDF 
model_tfidf = Sequential()
model_tfidf.add(Dense(64, activation='relu', input_dim=X_train_tfidf.shape[1]))
model_tfidf.add(Dropout(0.5))
model_tfidf.add(Dense(64, activation='relu'))
model_tfidf.add(Dropout(0.5))
model_tfidf.add(Dense(num_classes, activation='softmax'))
model_tfidf.compile(loss='categorical_crossentropy', optimizer='adam', 
metrics=['accuracy'])
model_tfidf.fit(X_train_tfidf.toarray(), y_train_onehot, epochs=10, 
batch_size=32, validation_data=(X_test_tfidf.toarray(), y_test_onehot))
y_pred_tfidf_nn = model_tfidf.predict(X_test_tfidf.toarray())
y_pred_tfidf_nn = np.argmax(y_pred_tfidf_nn, axis=1)
y_pred_tfidf_nn = label_encoder.inverse_transform(y_pred_tfidf_nn)
report_tfidf_nn = classification_report(y_test, y_pred_tfidf_nn)
print("Bag-of-Words Report with Neural Network:")
print(report_count_nn)
print("TF-IDF Report with Neural Network:")
print(report_tfidf_nn)
print("hello")