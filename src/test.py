import pandas as pd
import numpy as np
from numpy import triu
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
import re
from gensim.parsing.preprocessing import remove_stopwords
import joblib

tokenizer = joblib.load('artifacts/tokenizer.pkl')

test_text = pd.read_csv('dataset/test_dataset/test_text.csv')
test_labels = pd.read_csv('dataset/test_dataset/test_labels.csv')

def clean_text(sentences): 
    text = sentences.lower() 

    text = remove_stopwords(text)

    text = re.sub(r'[^\w\s]', '', text)

    text = re.sub('\w*\d\w*', '', text) 

    text = re.sub(r'\d', '', text)

    return text 

max_length = 120

cleaned_test_text = test_text.map(clean_text)

sequences_test = tokenizer.texts_to_sequences(cleaned_test_text['headline'])
test_padded = pad_sequences(sequences_test, padding='post', maxlen=max_length)

model = tf.keras.models.load_model('models/model.keras')

predictions = model.predict(test_padded)

predicted_labels = np.where(predictions > 0.5, 1, 0)

print("Confusion Matrix:")
print(confusion_matrix(test_labels, predicted_labels))
print("\nClassification Report:")
print(classification_report(test_labels, predicted_labels))
