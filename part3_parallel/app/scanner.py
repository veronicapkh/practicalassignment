"""поиск файлов и загрузка в массивы NumPy"""
import logging
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
TABLE_EXTENSIONS = {".csv"}
KIND_IMAGES = "images"
KIND_TABLES = "tables"
MAX_PIXEL_VALUE = 255.0

logger = logging.getLogger(__name__)


class PrepareError(Exception):
    """ошибка при поиске или чтении файлов"""


def parse_extensions(text):
    """преобразует строку с расширениями в набор
    параметры: text: расширения через запятую, с точкой или без
    возвращает: набор расширений в нижнем регистре"""
    result = set()
    for part in text.split(","):
        part = part.strip().lower()
        if part:
            result.add(part if part.startswith(".") else "." + part)
    if not result:
        raise PrepareError("не указано ни одного расширения")
    return result


def check_kind(extensions):
    """определяет, что загружается: картинки или таблицы
    параметры: extensions: набор расширений
    возвращает: KIND_IMAGES или KIND_TABLES"""
    unknown = extensions - IMAGE_EXTENSIONS - TABLE_EXTENSIONS
    if unknown:
        raise PrepareError(
            f"расширение не поддерживается: {', '.join(sorted(unknown))}.")
    if extensions <= IMAGE_EXTENSIONS:
        return KIND_IMAGES
    if extensions <= TABLE_EXTENSIONS:
        return KIND_TABLES
    raise PrepareError(
        "нельзя смешивать картинки и таблицы в одном запуске")


def find_files(root, extensions):
    """ищет файлы с нужными расширениями в папке
    и под-папках, сортирует по имени
    параметры: root: папка, с которой начинается поиск
               extensions: набор расширений
    возвращает: отсортированный список путей Path"""
    folder = Path(root)
    if not folder.is_dir():
        raise PrepareError(f"Папка не найдена: {folder}")
    return sorted(
        item for item in folder.rglob("*")
        if item.is_file() and item.suffix.lower() in extensions)


def load_image(path, size):
    """загружает картинку, меняет ее размер
    и приводит к массиву чисел от 0 до 1
    параметры: path: путь к картинке
               size: сторона квадрата в пикселях
    возвращает: массив (size, size, 3) с числами от 0 до 1"""
    try:
        with Image.open(path) as image:
            image = image.convert("RGB").resize((size, size))
            pixels = np.asarray(image, dtype=np.float32)
    except (UnidentifiedImageError, OSError) as error:
        raise PrepareError(
            f"не удалось прочитать картинку {path}: {error}") from error
    return pixels / MAX_PIXEL_VALUE


def load_table(path):
    """загружает данные из CSV-файла в массив чисел
    параметры: path: путь к CSV-файлу
    возвращает: двумерный массив чисел"""
    try:
        return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    except (ValueError, OSError) as error:
        raise PrepareError(
            f"не удалось прочитать таблицу {path}: {error}") from error


def load_files(files, kind, size):
    """Читает все файлы; битые пропускает и пишет предупреждение в лог.
    параметры: files: список путей
               kind: KIND_IMAGES или KIND_TABLES
               size: размер картинок (для таблиц не используется)
    возвращает: пару (список массивов, число пропущенных файлов)"""
    arrays = []
    skipped = 0
    for path in files:
        try:
            if kind == KIND_IMAGES:
                arrays.append(load_image(path, size))
            else:
                arrays.append(load_table(path))
        except PrepareError as error:
            logger.warning("файл пропущен: %s", error)
            skipped += 1
    if not arrays:
        raise PrepareError("не удалось прочитать ни одного файла")
    return arrays, skipped


def normalize_table(table):
    """приводит значения кажжого столбца к диапазону от 0 до 1
    параметры: table: двумерный массив чисел
    возвращает: нормализованный массив"""
    lowest = table.min(axis=0)
    spread = table.max(axis=0) - lowest
    spread[spread == 0] = 1.0
    return (table - lowest) / spread


def combine(arrays, kind):
    """объединяет данные из файлов в один массив
    параметры: arrays: список массивов
               kind: KIND_IMAGES или KIND_TABLES
    возвращает: общий массив: для картинок (число, size, size, 3),
                для таблиц (все строки, столбцы)"""
    if kind == KIND_IMAGES:
        return np.stack(arrays)
    try:
        table = np.vstack(arrays)
    except ValueError as error:
        raise PrepareError(
            f"у таблиц разное число столбцов: {error}") from error
    return normalize_table(table)
