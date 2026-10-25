"""измерение времени работы и сохранение отчета"""
import json
import time
from pathlib import Path


def measure(function, *args):
    """Запускает функцию и измеряет время ее работы
    параметры: function: функция для запуска
               args: её аргументы
    возвращает: пару (результат функции, время в секундах)"""
    started = time.perf_counter()
    result = function(*args)
    return result, time.perf_counter() - started


def build_report(details, sequential_seconds, parallel_seconds):
    """создает отчет и сравнивает время работы двух вариантов
    параметры: details: словарь с настройками запуска
               sequential_seconds: время последовательной версии
               parallel_seconds: время параллельной версии
    возвращает: словарь для записи в JSON"""
    report = dict(details)
    report["sequential_seconds"] = round(sequential_seconds, 3)
    report["parallel_seconds"] = round(parallel_seconds, 3)
    report["speedup"] = round(sequential_seconds / parallel_seconds, 2)
    return report


def save_report(report, path):
    """сохраняет отчет в JSON-файл
    параметры: report: словарь с результатами
               path: путь к файлу"""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    file_path.write_text(text, encoding="utf-8")
