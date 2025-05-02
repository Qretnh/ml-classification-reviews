from models.BAIES.model import NaiveBayesModel
from models.RNN.model import RNNModel
from models.kNN.model import kNNModel


def select_model(model_name) -> None | kNNModel | NaiveBayesModel | RNNModel:
    if model_name == "NaiveBayes":
        return NaiveBayesModel()
    elif model_name == "RNN":
        return RNNModel()
    elif model_name == "kNN":
        return kNNModel()