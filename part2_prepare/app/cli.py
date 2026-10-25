"""командная строка. Prepare и doctor"""
import argparse
import importlib
import logging
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
import numpy as np
from app.config import load_settings
from app.logging_config import setup_logging
from app.scanner import (PrepareError,
                         check_kind,
                         combine,
                         find_files,
                         load_files,
                         parse_extensions)

BYTES_IN_MB = 1024 * 1024
SUBPROCESS_TIMEOUT = 15
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
REQUIRED_LIBRARIES = {"numpy": "numpy",
                      "PIL": "pillow",
                      "dotenv": "python-dotenv"}

logger = logging.getLogger(__name__)


def build_parser(settings):
    """настривает команды и их аргументы
    параметры: settings: настройки (значения по умолчанию для аргументов)
    возвращает: объект ArgumentParser"""
    parser = argparse.ArgumentParser(
        description="подготовка данных для нейросети")
    parser.add_argument(
        "--log-level", choices=LOG_LEVELS, default=settings.log_level,
        help="уровень логирования (до команды)")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="собрать файлы из папки в один .npy")
    prepare.add_argument(
        "--path", default=str(settings.data_path),
        help="папка с файлами (ищем во всех вложенных тоже)")
    prepare.add_argument(
        "--ext", default=settings.extensions,
        help="расширения через запятую, например png,jpg или csv")
    prepare.add_argument(
        "--output", default=str(settings.output),
        help="куда сохранить результат (файл .npy)")
    prepare.add_argument(
        "--size", type=int, default=settings.image_size,
        help="сторона картинки в пикселях после уменьшения")
    commands.add_parser("doctor", help="проверить окружение")
    return parser


def run_prepare(args):
    """находит файлы, обрабатывает данные и сохраняет результат
    параметры: args: разобранные аргументы командной строки"""
    started = time.perf_counter()
    if args.size < 1:
        raise PrepareError("размер картинки должен быть больше 0")
    extensions = parse_extensions(args.ext)
    kind = check_kind(extensions)
    files = find_files(args.path, extensions)
    if not files:
        raise PrepareError("файлы с такими расширениями не найдены")
    logger.info("найдено файлов: %d", len(files))
    arrays, skipped = load_files(files, kind, args.size)
    data = combine(arrays, kind)
    output = Path(args.output).with_suffix(".npy")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        backup = output.with_name(output.name + ".bak")
        shutil.copy2(output, backup)
        logger.info("старый файл сохранен как %s", backup.name)
    np.save(output, data)
    elapsed = time.perf_counter() - started
    free_mb = shutil.disk_usage(output.parent).free / BYTES_IN_MB
    logger.info("прочитано файлов: %d, пропущено: %d",
                len(arrays), skipped)
    logger.info("размер массива: %s, тип: %s, в памяти: %.2f МБ",
                data.shape, data.dtype, data.nbytes / BYTES_IN_MB)
    logger.info("сохранено: %s (%.2f МБ)",
                output, output.stat().st_size / BYTES_IN_MB)
    logger.info("время обработки: %.3f с, свободно на диске: %.0f МБ",
                elapsed, free_mb)


def run_command(command, title):
    """запускает команду и выводит ее результат
    параметры: command: список слов команды
               title: подпись для вывода"""
    try:
        result = subprocess.run(
            command, capture_output=True, text=True,
            timeout=SUBPROCESS_TIMEOUT, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"{title}: не удалось запустить ({error})")
        return
    if result.returncode == 0:
        print(f"{title}: {result.stdout.strip()}")
    else:
        print(f"{title}: ошибка, код {result.returncode}, "
              f"{result.stderr.strip()}")


def run_doctor():
    """проверяет Python, библиотеки и видеокарту"""
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"рабочая папка: {Path.cwd()}")
    print("библиотеки:")
    for module_name, package_name in REQUIRED_LIBRARIES.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            print(f"  {package_name}: не найдена "
                  f"(pip install {package_name})")
        else:
            try:
                version = metadata.version(package_name)
            except metadata.PackageNotFoundError:
                version = "версия неизвестна"
            print(f"  {package_name}: найдена, {version}")
    run_command([sys.executable, "-m", "pip", "--version"], "pip")
    smi_path = shutil.which("nvidia-smi")
    if smi_path is None:
        print("CUDA: nvidia-smi не найден (видеокарта NVIDIA не обнаружена "
              "или драйвер не установлен)")
    else:
        run_command([smi_path, "--query-gpu=name,driver_version",
                     "--format=csv,noheader"], "CUDA (видеокарта, драйвер)")


def main(argv=None):
    """Точка входа: разбирает аргументы и запускает команду.

    параметры: argv: список аргументов (для тестов);
                     по умолчанию из командной строки
    возвращает: 0 при успехе, 1 при ошибке"""
    settings = load_settings()
    args = build_parser(settings).parse_args(argv)
    setup_logging(args.log_level, settings.log_dir)
    try:
        if args.command == "prepare":
            run_prepare(args)
        else:
            run_doctor()
    except (PrepareError, OSError) as error:
        logger.error("ошибка: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
