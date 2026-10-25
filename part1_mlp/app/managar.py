"""загрузка и обработка данных"""
from pathlib import Path
import numpy as np
from app.exceptions import (
    DataFileError,
    InvalidTrainingParamsError,
    MismatchedDataError,
    NetworkNotReadyError)


class DatasetManager:
    """загружает данные, нормализует их
    и делит на обучающую и тестовую части"""

    def __init__(self):
        """создает пустой объект для работы с данными"""
        self.features = None
        self.targets = None
        self.mean = None
        self.std = None
        self.train_features = None
        self.train_tarhets = None
        self.test_features = None
        self.test_targets = None

    def load_csv(self, path):
        """загружает данные из CSV-файла
        первая строка отображает название столбцов,
        последний стобец содержит правильные ответы"""
        file_path = Path(path)
        if not file_path.is_file():
            raise DataFileError(f"файл не найден:{file_path}")
        try:
            table = np.loadtxt(file_path, delimiter=",", skiprows=1, ndmin=2)
        except ValueError as error:
            raise DataFileError(f"неверный формат CSV: {error}") from error
        if table.size == 0:
            raise DataFileError("в файле нет данных.")
        if table.shape[1] < 2:
            raise MismatchedDataError(
                "нужно минимум 2 столбца: признаки и ответ.")
        self.features = table[:, :-1]
        self.targets = table[:, -1:]
        self.normalize()

    def normalize(self):
        """приводит признаки к одному масштабу"""
        self.mean = self.features.mean(axis=0)
        self.std = self.features.std(axis=0)
        self.std[self.std == 0] = 1.0
        self.features = (self.features - self.mean) / self.std

    def transform(self, row):
        """приводит новые данные к тому же масштабу, что и обучающие
        параметры: row: список значений признаков
        возвращает: нормализованные значения признаков"""
        if self.mean is None:
            raise NetworkNotReadyError("сначала загрузите данные из CSV.")
        values = np.asarray(row, dtype=float)
        if values.shape[-1] != self.mean.shape[0]:
            raise MismatchedDataError(
                f"нужно {self.mean.shape[0]} признаков на пример.")
        return (values - self.mean) / self.std

    def split(self, test_ratio):
        """делит данные на обучающую и тестовую часть"""
        if self.features is None:
            raise NetworkNotReadyError("Сначала загрузите данные из CSV.")
        if not 0 < test_ratio < 1:
            raise InvalidTrainingParamsError(
                "доля теста должна быть между 0 и 1.")
        order = np.random.default_rng().permutation(len(self.features))
        test_size = int(len(order) * test_ratio)
        if test_size == 0 or test_size == len(order):
            raise MismatchedDataError("Слишком мало строк для разбиения.")
        test_index = order[:test_size]
        train_index = order[test_size:]
        self.train_features = self.features[train_index]
        self.train_targets = self.targets[train_index]
        self.test_features = self.features[test_index]
        self.test_targets = self.targets[test_index]
