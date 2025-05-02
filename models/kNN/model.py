import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from typing import List, Dict

from sklearn.preprocessing import Normalizer

from exceptions.TrainingException import TrainingException
from exceptions.InternalException import InternalException

from sklearn.pipeline import Pipeline

from models.BaseModel import BaseModel

from transformers import AutoTokenizer, AutoModel
import torch

import string
import nltk
from nltk.corpus import stopwords



class BertEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self):
        model_name = "distilbert-base-uncased"
        nltk.download('stopwords')

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.get_bert_embeddings(X, self.model, self.tokenizer)

    def get_bert_embeddings(self, texts: List[str], model, tokenizer, batch_size=512):
        model.eval()
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = model(**inputs)

            last_hidden_states = outputs.last_hidden_state
            pooled = last_hidden_states[:, 0, :]
            embeddings.extend(pooled.cpu().numpy())
            print(i)
        return np.array(embeddings)



class kNNModel(BaseModel):
    """Модель на основе k-nearest-neighbours."""

    def __init__(self):
        self.MODEL_PATH = 'models/kNN/knn_model.joblib'
        self.TRAIN_DATASET_PATH = 'comments_dataset/youtubeCommentsDataset.csv'
        super().__init__()
        self.model = joblib.load(self.MODEL_PATH) if os.path.exists(self.MODEL_PATH) else None

    def learn(self, filepath: str = 'Default') -> str:
        if filepath == 'Default':
            filepath = self.TRAIN_DATASET_PATH
        if not filepath.endswith('.csv'):
            raise TrainingException("Wrong data for training.")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines: List[str] = f.read().splitlines()
            if lines[0].lower().startswith('text') and ',' in lines[0]:
                lines = lines[1:]
            data = []
            for line in lines:
                if ',' not in line:
                    continue  # Пропускаем некорректные строки
                parts = line.rsplit(',', 1)
                if len(parts) != 2:
                    continue
                text, sentiment = parts[0].strip(), parts[1].strip()
                text = self.preprocess(text)
                text = text.lower()
                data.append((text, sentiment))

            if not data:
                raise TrainingException("Missing data for training.")

            df = pd.DataFrame(data, columns=['text', 'sentiment'])

            min_count = df['sentiment'].value_counts().min()
            balanced_df = pd.concat([
                resample(df[df['sentiment'] == label],
                         replace=False,
                         n_samples=min_count,
                         random_state=42)
                for label in df['sentiment'].unique()
            ])
            df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

            X_train, y_train = df['text'], df['sentiment']

            pipeline = Pipeline([
                ('bert', BertEmbedder()),
                ('normalize', Normalizer()),
                ('knn', KNeighborsClassifier(
                    n_neighbors=9,
                    weights='distance',
                    metric='cosine',
                    algorithm='auto',
                ))
            ])

            pipeline.fit(X_train.tolist(), y_train.tolist())

            self.model = pipeline

            joblib.dump(pipeline, self.MODEL_PATH)
            return "Model trained and saved successfully."

        except Exception as e:

            raise InternalException(f"Error during training")


    def predict(self, file: bytes) -> Dict[str, str]:
        lines: List[str] = file.decode().splitlines()

        data = []
        for line in lines:
            if ',' not in line:
                continue  # пропускаем некорректные строки
            parts = line.rsplit(',', 1)
            if len(parts) != 2:
                continue
            text, sentiment = parts[0].strip(), parts[1].strip()
            data.append((text, sentiment))

        if not data:
            raise TrainingException("Missing data for testing.")

        df = pd.DataFrame(data, columns=['text', 'sentiment'])

        probas = self.model.predict_proba(df['text'].tolist())
        classes = self.model.classes_

        y_pred = []
        for prob in probas:
            max_prob = max(prob)
            max_class = classes[prob.argmax()]
            if max_class in ['positive', 'negative'] and max_prob < 0.36:
                y_pred.append('neutral')
            else:
                y_pred.append(max_class)
        y_true = df['sentiment']

        correct = (y_pred == y_true).sum()
        total = len(y_true)
        accuracy = round((correct / total) * 100, 2)
        failed = total - correct
        return {
            "accuracy": accuracy,
            "successfully_predicted": int(correct),
            "failed_to_predict": int(failed)
        }

    def preprocess(self, string_: str) -> str:
        STOPWORDS = set(stopwords.words('english'))
        text = string_.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
        text = re.sub(r'@\w+', '', text)  # remove mentions
        text = re.sub(r'#\w+', '', text)  # remove hashtags
        text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
        tokens = text.split()
        tokens = [word for word in tokens if word not in STOPWORDS]
        return ' '.join(tokens)

