"""обработка картинок с помощью потоков и процессов"""
import logging
import multiprocessing
import queue
import threading
from pathlib import Path
import numpy as np
from app.augment import augment_image
from app.scanner import PrepareError, find_files, load_image

STOP_SIGNAL = None
QUEUE_SIZE = 50
QUEUE_TIMEOUT = 0.5
PROGRESS_STEP = 50

logger = logging.getLogger(__name__)


class SafeCounter:
    """счетчик, который можно использовать в нескольких потоках"""

    def __init__(self):
        """создает счетчик со значением ноль"""
        self._value = 0
        self._lock = threading.Lock()

    def increment(self):
        """увеличивает счетчик на 1 и возвращает новое значение
        возвращает: новое значение счётчика"""
        with self._lock:
            self._value += 1
            return self._value

    @property
    def value(self):
        """возвращает текущее значение счетчика"""
        with self._lock:
            return self._value


def put_item(target_queue, item, stop_event):
    """кладет элемент в очередь, пока программа не остановлена
    параметры: target_queue: очередь queue.Queue
               item: что положить
               stop_event: событие остановки
    возвращает: True, если положили; False, если пришла остановка"""
    while not stop_event.is_set():
        try:
            target_queue.put(item, timeout=QUEUE_TIMEOUT)
            return True
        except queue.Full:
            continue
    return False


def get_item(source_queue, stop_event):
    """берет элемент из очереди, пока программа не остановлена
    параметры: source_queue: очередь queue.Queue
               stop_event: событие остановки
    возвращает: элемент или STOP_SIGNAL, если пришла остановка"""
    while not stop_event.is_set():
        try:
            return source_queue.get(timeout=QUEUE_TIMEOUT)
        except queue.Empty:
            continue
    return STOP_SIGNAL


def producer(root, extensions, path_queue, consumers_count, stop_event):
    """ищет картинки и передает их пути в очередь
    параметры: root: папка для поиска
               extensions: набор расширений
               path_queue: очередь путей
               consumers_count: сколько потоков-потребителей ждут сигнала конца
               stop_event: событие остановки"""
    logger.info("Производитель: начал сканирование %s", root)
    found = 0
    try:
        for item in sorted(Path(root).rglob("*")):
            if item.is_file() and item.suffix.lower() in extensions:
                if not put_item(path_queue, item, stop_event):
                    return
                found += 1
        logger.info("производитель: нашел файлов %d", found)
    finally:
        for _ in range(consumers_count):
            put_item(path_queue, STOP_SIGNAL, stop_event)


def consumer(path_queue, array_queue, size, loaded, skipped, stop_event):
    """читает картинки и передает их массивы в другую очередь
    параметры: path_queue: очередь путей (откуда берём)
               array_queue: очередь массивов (куда кладём)
               size: сторона картинки в пикселях
               loaded: общий счётчик прочитанных файлов
               skipped: общий счётчик пропущенных файлов
               stop_event: событие остановки"""
    handled = 0
    try:
        while True:
            path = get_item(path_queue, stop_event)
            if path is STOP_SIGNAL:
                break
            try:
                array = load_image(path, size)
            except PrepareError as error:
                logger.warning("файл пропущен: %s", error)
                skipped.increment()
                continue
            index = loaded.increment()
            handled += 1
            logger.debug("прочитан %s (номер %d)", path.name, index)
            if index % PROGRESS_STEP == 0:
                logger.info("прочитано файлов: %d", index)
            if not put_item(array_queue, (index, array), stop_event):
                break
    finally:
        put_item(array_queue, STOP_SIGNAL, stop_event)
        logger.info("потребитель: закончил, обработал файлов %d", handled)


def run_sequential(root, extensions, size, copies):
    """обрабатывает картинки по очереди, без параллельной работы
    параметры: root: папка с картинками
               extensions: набор расширений
               size: сторона картинки в пикселях
               copies: сколько копий делать из каждой картинки
    возвращает: пару (массив всех картинок, словарь loaded/skipped)"""
    originals = []
    augmented = []
    skipped = 0
    for index, path in enumerate(find_files(root, extensions), start=1):
        try:
            array = load_image(path, size)
        except PrepareError as error:
            logger.warning("файл пропущен: %s", error)
            skipped += 1
            continue
        originals.append(array)
        augmented.extend(augment_image(array, copies, index))
    if not originals:
        raise PrepareError("не удалось прочитать ни одной картинки")
    stats = {"loaded": len(originals), "skipped": skipped}
    return np.stack(originals + augmented), stats


def run_parallel(root, extensions, size, copies, thread_count,
                 process_count):
    """обрабатывает картинки параллельно с помощью потоков и процессов
    параметры: root: папка с картинками
               extensions: набор расширений
               size: сторона картинки в пикселях
               copies: сколько копий делать из каждой картинки
               thread_count: число потоков-потребителей
               process_count: число процессов в пуле
    возвращает: пару (массив всех картинок, словарь loaded/skipped)"""
    path_queue = queue.Queue(maxsize=QUEUE_SIZE)
    array_queue = queue.Queue(maxsize=QUEUE_SIZE)
    stop_event = threading.Event()
    loaded = SafeCounter()
    skipped = SafeCounter()
    results = []
    results_lock = threading.Lock()
    originals = []
    pending = []

    def collect(copies_list):
        """добавляет готовые копии в общий список"""
        with results_lock:
            results.append(copies_list)

    def report_error(error):
        """записывает ошибку обработки в журнал"""
        logger.error("Ошибка в процессе: %s", error)

    threads = [threading.Thread(
        target=producer, name="producer",
        args=(root, extensions, path_queue, thread_count, stop_event))]
    for number in range(1, thread_count + 1):
        threads.append(threading.Thread(
            target=consumer, name=f"consumer-{number}",
            args=(path_queue, array_queue, size, loaded, skipped,
                  stop_event)))
    context = multiprocessing.get_context("spawn")
    pool = context.Pool(processes=process_count)
    try:
        for thread in threads:
            thread.start()
        finished = 0
        while finished < thread_count:
            item = get_item(array_queue, stop_event)
            if item is STOP_SIGNAL:
                finished += 1
                continue
            index, array = item
            originals.append(array)
            pending.append(pool.apply_async(
                augment_image, (array, copies, index),
                callback=collect, error_callback=report_error))
        for async_result in pending:
            async_result.get()
        pool.close()
    except BaseException:
        stop_event.set()
        pool.terminate()
        raise
    finally:
        for thread in threads:
            thread.join()
        pool.join()
    if not originals:
        raise PrepareError("не удалось прочитать ни одной картинки")
    augmented = [array for group in results for array in group]
    stats = {"loaded": loaded.value, "skipped": skipped.value}
    return np.stack(originals + augmented), stats
