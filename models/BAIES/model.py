import os
import joblib
import pandas as pd
from typing import List, Dict

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample

from models.BaseModel import BaseModel

from exceptions.TrainingException import TrainingException
from exceptions.InternalException import InternalException

import re

import emoji

import nltk
from nltk.stem import WordNetLemmatizer


class NaiveBayesModel(BaseModel):

    def __init__(self):
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()
        self.MODEL_PATH = 'models/BAIES/nb_model.joblib'
        self.TRAIN_DATASET_PATH = 'comments_dataset/youtubeCommentsDataset.csv'
        self.model = joblib.load(self.MODEL_PATH) if os.path.exists(self.MODEL_PATH) else None
        super().__init__()

    def learn(self, filepath: str = "Default") -> str:
        if filepath == "Default":
            filepath = self.TRAIN_DATASET_PATH
        if not filepath.endswith('.csv'):
            raise TrainingException("Wrong data format. Only .csv is supported.")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines: List[str] = f.read().splitlines()
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

            df_major = df[df['sentiment'] == 'positive']
            df_medium = df[df['sentiment'] == 'neutral']
            df_minor = df[df['sentiment'] == 'negative']

            df_negative_up = resample(df_minor, replace=True, n_samples=len(df_major), random_state=42)
            df_neutral_up = resample(df_medium, replace=True, n_samples=len(df_major), random_state=42)

            df = pd.concat([df_major, df_neutral_up, df_negative_up])

            df = df.sample(frac=1, random_state=42)  # перемешать

            X_train, y_train = df['text'], df['sentiment']

            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=30000,
                    ngram_range=(1, 4),
                    analyzer='word',
                    norm='l2',
                    max_df=0.8,
                    sublinear_tf=True,
                )),
                ('nb', MultinomialNB(
                    alpha=0.95,
                    fit_prior=False,))
            ])

            pipeline.fit(X_train, y_train)

            self.model = pipeline
            joblib.dump(pipeline, self.MODEL_PATH)

            return "Model trained and saved successfully."

        except Exception as e:
            raise TrainingException(f"Error during prediction: {str(e)}")

    def predict(self, file: bytes) -> Dict[str, str]:
        lines: List[str] = file.decode().splitlines()
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
            data.append((text, sentiment))

        if not data:
            raise TrainingException("Missing data for testing.")

        df = pd.DataFrame(data, columns=['text', 'sentiment'])

        if not self.model:
            raise InternalException("Model is not trained or failed to load.")

        probas = self.model.predict_proba(df['text'])
        classes = self.model.classes_

        y_pred = []
        for prob in probas:
            max_prob = max(prob)
            max_class = classes[prob.argmax()]
            if max_class in ['positive', 'negative'] and max_prob < 0.40:
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
        text = string_.lower()
        text = emoji.replace_emoji(text, '')  # убрать эмодзи
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r"[^a-zA-Z\s]", '', text)
        text = re.sub(r'\b(.)\1{2,}\b', r'\1', text)  # loooove -> lo
        text = re.sub(r"\s+", ' ', text).strip()
        text = ' '.join([self.lemmatizer.lemmatize(word) for word in text.split()])
        return text
