"""изменение картинок: повороты, отражения, размытие, шум"""
import numpy as np

NOISE_LEVEL = 0.05
MAX_ROTATIONS = 4
BLUR_RADIUS = 2


def blur(image, radius):
    """Размывает картинку и усредняет значения соседних пикселей
    параметры: image: массив (высота, ширина, 3)
               radius: на сколько пикселей смотрим вокруг
    возвращает: размытый массив того же размера"""
    total = np.zeros_like(image)
    for shift_y in range(-radius, radius + 1):
        for shift_x in range(-radius, radius + 1):
            total += np.roll(image, (shift_y, shift_x), axis=(0, 1))
    return total / (2 * radius + 1) ** 2


def augment_once(image, generator):
    """создает одну измененную копию картинки
    параметры: image: массив (высота, ширина, 3) с числами от 0 до 1
               generator: генератор случайных чисел NumPy
    возвращает: новый массив того же размера"""
    result = np.rot90(image, k=int(generator.integers(0, MAX_ROTATIONS)))
    if generator.random() < 0.5:
        result = np.fliplr(result)
    if generator.random() < 0.5:
        result = np.flipud(result)
    result = blur(result, BLUR_RADIUS)
    noise = generator.normal(0.0, NOISE_LEVEL, result.shape)
    return np.clip(result + noise, 0.0, 1.0).astype(np.float32)


def augment_image(image, copies, seed):
    """создает несколько измененных копий картики
    параметры: image: массив (высота, ширина, 3) с числами от 0 до 1
               copies: сколько копий сделать
               seed: число для генератора случайных чисел
    возвращает: список массивов-копий"""
    generator = np.random.default_rng(seed)
    return [augment_once(image, generator) for _ in range(copies)]
