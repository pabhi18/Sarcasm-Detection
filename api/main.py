from fastapi import FastAPI
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from gensim.parsing.preprocessing import remove_stopwords
import joblib
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)

tokenizer = joblib.load('artifacts/tokenizer.pkl')
model = tf.keras.models.load_model('models/model.keras')

class InputText(BaseModel):
    text: str

def clean_text(sentence): 
    text = sentence.lower() 

    text = remove_stopwords(text)

    text = re.sub(r'[^\w\s]', '', text)

    text = re.sub('\w*\d\w*', '', text) 

    text = re.sub(r'\d', '', text)

    return text

@app.post("/")
def predict_model(input_text: InputText):
    cleaned_text = clean_text(input_text.text)
    sequences_text = tokenizer.texts_to_sequences([cleaned_text])
    padded_text = pad_sequences(sequences_text, padding='post', maxlen=120)

    prediction = model.predict(padded_text)

    if prediction > 0.5:
        return {'result': 'Sarcastic Sentence'}
    else:
        return {'result': 'Not a Sarcastic Sentence'}