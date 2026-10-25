"""настройка логирования"""
import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
LOG_FILE_NAME = "prepare.log"


def setup_logging(level_name, log_dir):
    """настроивает вывод сообщений в консоль и файл
    параметры: level_name: уровень, например "INFO" или "DEBUG"
               log_dir: папка для файла журнала"""
    level = logging.getLevelNamesMapping().get(
        str(level_name).upper(), logging.INFO)
    folder = Path(log_dir)
    folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=level,
                        format=LOG_FORMAT,
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(folder / LOG_FILE_NAME,
                                                      encoding="utf-8")],
                        force=True)
