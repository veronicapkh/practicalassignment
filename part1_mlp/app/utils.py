"""функции для ввода данных через консоль"""
from pathlib import Path
import numpy as np
from app.exceptions import (DataFileError,
                            InvalidLayerSizeError,
                            InvalidTrainingParamsError)


def clean_path(text):
    """убирает кавычки и пробелы в начале и конце строки
    параметры: text: строка, введенная пользователяем
    возвращает: строка без кавычек и пробелов"""
    return text.strip().strip('"').strip("'")


def ask_float(question, default=None):
    """запрашивает число с плавающей точкой у пользователя
    парметры: question: текст вопроса
              default: значение по умолчанию, если пользователь ничего не ввел
    возвращает: число с плавающей точкой"""
    text = input(question).strip()
    if text == "" and default is not None:
        return default
    try:
        return float(text.replace(",", "."))
    except ValueError as error:
        raise InvalidTrainingParamsError(
            f"нужно ввести число, а введено:'{text}'.") from error


def ask_int(question, default=None):
    """запрашивает целое число у пользователя
    параметры: question: текст вопроса
              default: значение по умолчанию, если пользователь ничего не ввел
    возвращает: целое число"""
    text = input(question).strip()
    if text == "" and default is not None:
        return default
    try:
        return int(text)
    except ValueError as error:
        raise InvalidTrainingParamsError(
            f"нужно ввести целое число, а введено:'{text}'.") from error


def ask_layer_sizes(question):
    """запрашивает у пользователя размеры слоев сети
    параметры: question: текст вопроса
    возвращает: список целых чисел"""
    text = input(question).strip()
    try:
        return [int(part) for part in text.split(",")]
    except ValueError as error:
        raise InvalidLayerSizeError(
            f"нужно ввести размеры слоев сети, а введено:'{text}'.") from error


def ask_numbers(question):
    """запрашивает у пользователя список чисел, разделенных пробелами
    параметры: question: текст вопроса
    возвращает: список чисел с плавающей точкой"""
    text = input(question).strip()
    try:
        return [float(part.replace(",", ".")) for part in text.split()]
    except ValueError as error:
        raise InvalidTrainingParamsError(
            "нужно ввести числа, разделенные пробелами") from error


def load_rows(path):
    """читает данные из CSV-файла и возвращает их в виде массива numpy
    параметры: path: путь к CSV-файлу
    возвращает: массив numpy с данными"""
    file_path = Path(path)
    if not file_path.is_file():
        raise DataFileError(f"файл не найден:{file_path}")
    try:
        return np.loadtxt(file_path, delimiter=",", skiprows=1, ndmin=2)
    except ValueError as error:
        raise DataFileError(f"неверный формат файла: {error}") from error
