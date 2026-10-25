"""чтение картинок в потоках и обработка в процессах"""
import argparse
import logging
import sys
import numpy as np
from app.config import load_settings
from app.logging_config import setup_logging
from app.performance import build_report, measure, save_report
from app.producer_consumer import run_parallel, run_sequential
from app.scanner import KIND_IMAGES, PrepareError, check_kind, parse_extensions

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

logger = logging.getLogger(__name__)


def build_parser(settings):
    """настраивает команды и их аргументы
    параметры: settings: настройки (значения по умолчанию для аргументов)
    возвращает: объект ArgumentParser"""
    parser = argparse.ArgumentParser(
        description="аугментация картинок: потоки читают, процессы "
                    "обрабатывают; сравнение с последовательной версией")
    parser.add_argument("--path", default=str(settings.data_path),
                        help="папка с картинками (ищем и во вложенных)")
    parser.add_argument("--ext", default=settings.extensions,
                        help="расширения через запятую, например png,jpg")
    parser.add_argument("--size", type=int, default=settings.image_size,
                        help="сторона картинки в пикселях")
    parser.add_argument("--copies", type=int, default=settings.copies,
                        help="сколько копий делать из каждой картинки")
    parser.add_argument("--threads", type=int, default=settings.threads,
                        help="число потоков, читающих файлы")
    parser.add_argument("--processes", type=int, default=settings.processes,
                        help="число процессов для аугментации")
    parser.add_argument("--report", default=str(settings.report),
                        help="куда записать performance_report.json")
    parser.add_argument("--output", default=settings.output,
                        help="куда сохранить набор данных .npy "
                             "(по умолчанию не сохраняем)")
    parser.add_argument("--log-level", choices=LOG_LEVELS,
                        default=settings.log_level,
                        help="уровень логирования")
    return parser


def check_arguments(args):
    """проверяет, что числовые аргументы больше нуля
    параметры: args: разобранные аргументы"""
    for name in ("size", "copies", "threads", "processes"):
        if getattr(args, name) < 1:
            raise PrepareError(f"аргумент --{name} должен быть больше нуля")


def run(args):
    """Запускает обе версии обработки, сравнивает время и пишет отчёт
    параметры: args: разобранные аргументы командной строки"""
    check_arguments(args)
    extensions = parse_extensions(args.ext)
    if check_kind(extensions) != KIND_IMAGES:
        raise PrepareError("аугментация работает только с картинками")
    logger.info("последовательная версия: старт")
    (sequential_data, _), sequential_time = measure(
        run_sequential, args.path, extensions, args.size, args.copies)
    sequential_shape = sequential_data.shape
    del sequential_data
    logger.info("последовательная версия: %.3f с", sequential_time)
    logger.info("параллельная версия: старт (потоков %d, процессов %d)",
                args.threads, args.processes)
    (parallel_data, stats), parallel_time = measure(
        run_parallel, args.path, extensions, args.size, args.copies,
        args.threads, args.processes)
    logger.info("параллельная версия: %.3f с", parallel_time)
    if sequential_shape != parallel_data.shape:
        logger.warning("размеры наборов не совпали: %s и %s",
                       sequential_shape, parallel_data.shape)
    details = {
        "images_loaded": stats["loaded"],
        "images_skipped": stats["skipped"],
        "image_size": args.size,
        "copies_per_image": args.copies,
        "reader_threads": args.threads,
        "worker_processes": args.processes,
        "dataset_shape": list(parallel_data.shape),
    }
    report = build_report(details, sequential_time, parallel_time)
    save_report(report, args.report)
    logger.info("набор данных: %s; ускорение: x%.2f; отчёт: %s",
                parallel_data.shape, report["speedup"], args.report)
    if args.output:
        np.save(args.output, parallel_data)
        logger.info("набор данных сохранен: %s", args.output)


def main(argv=None):
    """запускает программу и обрабатывает ошибки
    параметры: argv: список аргументов (для тестов);
                     по умолчанию из командной строки
    возвращает: 0 при успехе, 1 при ошибке, 130 при остановке по Ctrl+C"""
    settings = load_settings()
    args = build_parser(settings).parse_args(argv)
    setup_logging(args.log_level, settings.log_dir)
    try:
        run(args)
    except KeyboardInterrupt:
        logger.warning("остановлено пользователем (Ctrl+C)")
        return 130
    except (PrepareError, OSError) as error:
        logger.error("ошибка: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
