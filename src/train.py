import pandas as pd
import tensorflow as tf 
from gensim.parsing.preprocessing import remove_stopwords
import re
import string
import numpy as np 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import train_test_split
import joblib
import mlflow
from mlflow.tensorflow import MlflowCallback

mlflow.login()


df1 = pd.read_json('dataset/Sarcasm_Headlines_Dataset.json', lines=True)
df2 = pd.read_json('dataset/Sarcasm_Headlines_Dataset_v2.json', lines=True)

df = pd.concat([df1, df2], ignore_index=True) 

def clean_text(sentences): 
    text = sentences.lower() 

    text = remove_stopwords(text)

    text = re.sub(r'[^\w\s]', '', text)

    text = re.sub('\w*\d\w*', '', text) 

    text = re.sub(r'\d', '', text)

    return text 

df['cleaned_headline']=df['headline'].map(clean_text)

X = df['cleaned_headline']
y = df['is_sarcastic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

max_length = 120

vocab_size = 10000

embedding_dim = 100

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>') 

tokenizer.fit_on_texts(X_train)

sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_valid = tokenizer.texts_to_sequences(X_val)

train_padded = pad_sequences(sequences_train, 
                             padding='post', 
                             maxlen=max_length) 

val_padded = pad_sequences(sequences_valid, 
                             padding='post', 
                             maxlen=max_length) 

model = tf.keras.Sequential([ 
    tf.keras.layers.Input(shape=(120,)),
    tf.keras.layers.Embedding( 
        input_dim = vocab_size, output_dim = embedding_dim),
  
    tf.keras.layers.LSTM(32, return_sequences = True), 

    tf.keras.layers.Flatten(), 

    tf.keras.layers.Dropout(0.5), 

    tf.keras.layers.Dense(24, activation='relu'), 
  
    tf.keras.layers.Dropout(0.2), 
  
    tf.keras.layers.Dense(1, activation='sigmoid') 
]) 
  
model.summary() 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

mlflow.set_experiment("/Sarcasm-Detection")
mlflow.tensorflow.autolog(disable=True)

with mlflow.start_run() as run:
    model.fit(train_padded, 
              y_train, 
              epochs=5, 
              validation_data=(val_padded, 
              y_val),
              callbacks=[MlflowCallback(run)])

    model.save('models/model.keras')