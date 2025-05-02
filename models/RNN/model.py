import os
import re
import joblib

import pandas as pd

from typing import List, Dict

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical

from models.BaseModel import BaseModel

from exceptions.TrainingException import TrainingException
from exceptions.InternalException import InternalException

import nltk
from nltk.stem import WordNetLemmatizer

import emoji



class RNNModel(BaseModel):
    def __init__(self):
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()

        self.MODEL_PATH = 'models/RNN/rnn_model.h5'
        self.TOKENIZER_PATH = 'models/RNN/tokenizer.pkl'
        self.MAX_LEN = 100
        super().__init__()
        self.model = load_model(self.MODEL_PATH) if os.path.exists(self.MODEL_PATH) else None
        self.tokenizer = joblib.load(self.TOKENIZER_PATH) if os.path.exists(self.TOKENIZER_PATH) else None

    def learn(self, filepath: str = 'comments_dataset/youtubeCommentsDataset.csv') -> str:
        if not filepath.endswith('.csv'):
            raise TrainingException("Wrong data format. Must be CSV.")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
            if lines[0].lower().startswith('text') and ',' in lines[0]:
                lines = lines[1:]
            data = []
            for line in lines:
                if ',' not in line:
                    continue
                parts = line.rsplit(',', 1)
                if len(parts) != 2:
                    continue
                text, sentiment = parts[0].strip(), parts[1].strip()
                text = self.preprocess(text)
                data.append((text, sentiment))

            if not data:
                raise TrainingException("Missing data for training.")

            df = pd.DataFrame(data, columns=['text', 'sentiment'])

            self.tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
            self.tokenizer.fit_on_texts(df['text'])

            sequences = self.tokenizer.texts_to_sequences(df['text'])
            padded = pad_sequences(sequences, maxlen=self.MAX_LEN, padding='post', truncating='post')
            labels = pd.get_dummies(df['sentiment']).values

            self.model = Sequential([
                Embedding(input_dim=10000, output_dim=64),
                Bidirectional(LSTM(64)),
                Dropout(0.5),
                Dense(labels.shape[1], activation='softmax')
            ])

            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.model.fit(padded, labels, epochs=5, batch_size=64, verbose=1)

            self.model.save(self.MODEL_PATH)
            joblib.dump(self.tokenizer, self.TOKENIZER_PATH)

            return "RNN model trained and saved successfully."

        except Exception as e:
            raise InternalException(f"Error during training: {str(e)}")

    def predict(self, file: bytes) -> Dict[str, str]:
        try:
            lines = file.decode().splitlines()
            if lines[0].lower().startswith('text') and ',' in lines[0]:
                lines = lines[1:]
            data = []
            for line in lines:
                if ',' not in line:
                    continue
                parts = line.rsplit(',', 1)
                if len(parts) != 2:
                    continue
                text, sentiment = parts[0].strip(), parts[1].strip()
                text = self.preprocess(text)
                data.append((text, sentiment))

            if not data:
                raise TrainingException("Missing data for prediction.")

            df = pd.DataFrame(data, columns=['text', 'sentiment'])

            sequences = self.tokenizer.texts_to_sequences(df['text'])
            padded = pad_sequences(sequences, maxlen=self.MAX_LEN, padding='post', truncating='post')

            predictions = self.model.predict(padded)

            label_columns = list(pd.get_dummies(df['sentiment']).columns)
            predicted_labels = pd.DataFrame(predictions, columns=label_columns).idxmax(axis=1)

            correct = (predicted_labels.values == df['sentiment'].values).sum()
            total = len(df)
            accuracy = round((correct / total) * 100, 2)
            failed = total - correct

            return {
                "accuracy": accuracy,
                "successfully_predicted": int(correct),
                "failed_to_predict": int(failed)
            }

        except Exception as e:
            raise InternalException(f"Error during prediction: {str(e)}")

    def preprocess(self, string: str) -> str:
        text = string.lower()
        text = emoji.replace_emoji(text, '')  # убрать эмодзи
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r"[^a-zA-Z\s]", '', text)
        text = re.sub(r'\b(.)\1{2,}\b', r'\1', text)  # loooove -> lo
        text = re.sub(r"\s+", ' ', text).strip()
        text = ' '.join([self.lemmatizer.lemmatize(word) for word in text.split()])
        return text
