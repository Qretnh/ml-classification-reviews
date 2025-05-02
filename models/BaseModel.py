from abc import ABC

from typing import Dict


class BaseModel(ABC):
    """Абстрактный базовый класс для моделей машинного обучения.
    Наследники должны реализовать методы learn() и predict().
    """
    def __init__(self) -> None:
        """Создаётся объект модели с указанным названием файла дампа модели."""
        pass

    def learn(self, filepath: str) -> str:
        """Обучает модель на данных из filepath.
        Args:
            filepath: Путь к файлу с данными.
        Returns:
            Сообщение о результате обучения.
        """
        pass


    def predict(self, file: bytes) -> Dict[str, str]:
        """Предсказывает результат на входных данных.
        Args:
            file: Входные данные в байтах.
        Returns:
            Словарь с результатами (например, accuracy).
        """
        pass