"""настройки программы"""
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_PATH = "data/input"
DEFAULT_EXTENSIONS = "png,jpg"
DEFAULT_OUTPUT = "data/output/dataset.npy"
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_IMAGE_SIZE = 32


@dataclass
class Settings:
    """хранит настройки программы"""
    data_path: Path
    extensions: str
    output: Path
    log_dir: Path
    log_level: str
    image_size: int


def resolve_path(text):
    """преобразует путь в полный путь к файлу или папке
    параметры: text: путь строкой
    возвращает: объект Path с полным путем"""
    path = Path(text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_settings():
    """загружает настройки из файла .env и возвращает объект Settings"""
    load_dotenv(PROJECT_ROOT / ".env")
    size_text = os.getenv("IMAGE_SIZE", str(DEFAULT_IMAGE_SIZE))
    image_size = int(size_text) if size_text.isdigit() else DEFAULT_IMAGE_SIZE
    return Settings(
        data_path=resolve_path(os.getenv("DATA_PATH", DEFAULT_DATA_PATH)),
        extensions=os.getenv("EXTENSIONS", DEFAULT_EXTENSIONS),
        output=resolve_path(os.getenv("OUTPUT_FILE", DEFAULT_OUTPUT)),
        log_dir=resolve_path(os.getenv("LOG_DIR", DEFAULT_LOG_DIR)),
        log_level=os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL),
        image_size=image_size)
