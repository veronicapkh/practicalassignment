"""настройки программы"""
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = "data/perf_images"
DEFAULT_EXTENSIONS = "png,jpg"
DEFAULT_OUTPUT = ""
DEFAULT_REPORT = "data/output/performance_report.json"
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_IMAGE_SIZE = 96
DEFAULT_COPIES = 6
DEFAULT_THREADS = 4


@dataclass
class Settings:
    """хранит настройки программы"""
    data_path: Path
    extensions: str
    output: str
    report: Path
    log_dir: Path
    log_level: str
    image_size: int
    copies: int
    threads: int
    processes: int


def resolve_path(text):
    """хранит настройки программы
    параметры: text: путь строкой
    возвращает: объект Path с полным путем"""
    path = Path(text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_int(name, default):
    """читает целое число из настроек
    параметры: name: имя переменной
               default: значение, если переменной нет или в ней не число
    возвращает: целое число"""
    text = os.getenv(name, "")
    return int(text) if text.isdigit() else default


def load_settings():
    """загружает настрйоки из .env и использует значения по умолчанию
    возвращает: объект Settings"""
    load_dotenv(PROJECT_ROOT / ".env")
    output = os.getenv("AUGMENTED_FILE", DEFAULT_OUTPUT)
    return Settings(
        data_path=resolve_path(os.getenv("DATA_PATH", DEFAULT_DATA_PATH)),
        extensions=os.getenv("EXTENSIONS", DEFAULT_EXTENSIONS),
        output=str(resolve_path(output)) if output else "",
        report=resolve_path(os.getenv("REPORT_FILE", DEFAULT_REPORT)),
        log_dir=resolve_path(os.getenv("LOG_DIR", DEFAULT_LOG_DIR)),
        log_level=os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL),
        image_size=read_int("IMAGE_SIZE", DEFAULT_IMAGE_SIZE),
        copies=read_int("COPIES", DEFAULT_COPIES),
        threads=read_int("THREADS", DEFAULT_THREADS),
        processes=read_int("PROCESSES", os.cpu_count() or 2))
