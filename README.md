# Sentiment Analysis of Movie Reviews

## Objective

The goal of this project is to build and evaluate three models for sentiment analysis of movie reviews using Naive Bayes, Logistic Regression, and a neural network. Python libraries utilized include Numpy, pandas, sklearn, keras, and NLTK.

## Data

The dataset used for this project consists of text reviews and their corresponding sentiment labels (positive or negative). You can access the dataset from the IMDB movie reviews dataset available on Kaggle:
[IMDB Movie Reviews Dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Method

### Data Preprocessing

- Import the dataset and perform basic data cleaning and preprocessing. This includes removing stop words, punctuation, and stemming the text.
- Split the dataset into training and test sets.

### Feature Extraction

- Convert the text data into numerical features using techniques such as bag of words or TF-IDF.

### Model Building

- Train a Naive Bayes classifier on the extracted features using a training algorithm provided by the NLTK library.
- Train a logistic regression model on the extracted features using a suitable training algorithm.
- Train a simple neural network model with one hidden layer on the extracted features using a suitable training algorithm.

### Model Evaluation

- Evaluate the performance of the trained Naive Bayes model on the test dataset using metrics such as accuracy, precision, recall, and F1 score.
- Evaluate the performance of the trained logistic regression model on the test dataset using metrics such as accuracy, precision, recall, and F1 score.
- Evaluate the performance of the trained neural network model on the test dataset using metrics such as accuracy, precision, recall, and F1 score.

### Model Comparison

- Compare the performance of the three models and discuss which one performed better and why.

### Deliverables

- A report detailing your approach and findings, including code and relevant visualizations.

