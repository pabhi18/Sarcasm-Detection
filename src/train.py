import pandas as pd
import tensorflow as tf 
import re
from gensim.parsing.preprocessing import remove_stopwords
import numpy as np 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from mlflow.tensorflow import MlflowCallback
import joblib
import mlflow

mlflow.login()

train_text = pd.read_csv('dataset/train_dataset/train_text.csv')
train_labels = pd.read_csv('dataset/train_dataset/train_labels.csv')
valid_text = pd.read_csv('dataset/valid_dataset/valid_text.csv')
valid_labels = pd.read_csv('dataset/valid_dataset/valid_labels.csv')

def clean_text(sentences): 
    text = sentences.lower() 

    text = remove_stopwords(text)

    text = re.sub(r'[^\w\s]', '', text)

    text = re.sub('\w*\d\w*', '', text) 

    text = re.sub(r'\d', '', text)

    return text 

cleaned_train_text = train_text.map(clean_text)
cleaned_valid_text = valid_text.map(clean_text)

max_length = 120

vocab_size = 10000

embedding_dim = 100

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>') 
tokenizer.fit_on_texts(cleaned_train_text['headline'])

joblib.dump(tokenizer, 'artifacts/tokenizer.pkl')

sequences_train = tokenizer.texts_to_sequences(cleaned_train_text['headline'])
sequences_valid = tokenizer.texts_to_sequences(cleaned_valid_text['headline'])

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
              train_labels, 
              epochs=5, 
              validation_data=(val_padded, 
              valid_labels),
              callbacks=[MlflowCallback(run)])

    model.save('models/model.keras')