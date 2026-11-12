"""Препроцессинг изображений документа (OpenCV).

Реализованы алгоритмы из ТЗ:
* определение и компенсация наклона (deskew) через minAreaRect / HoughLinesP;
* удаление синих и фиолетовых печатей (HSV-маска) до бинаризации;
* шумоподавление (fastNlMeansDenoising) и адаптивная бинаризация Гаусса.

Функции написаны так, чтобы модуль работал и без установленного OpenCV
(возвращается исходное изображение) — это удобно для лёгких CPU-контуров.
"""
from __future__ import annotations

import logging

import numpy as np

from app.core.profiling import stage

logger = logging.getLogger(__name__)

try:  # pragma: no cover - зависит от окружения
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if cv2 is not None else image


def remove_stamps(image: np.ndarray) -> np.ndarray:
    """Заменяет синие/фиолетовые пиксели печатей на белый фон."""
    if cv2 is None:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # два диапазона: синий (~100-130) и фиолетовый/пурпурный (~130-165)
    mask_blue = cv2.inRange(hsv, np.array([90, 45, 40]), np.array([135, 255, 255]))
    mask_violet = cv2.inRange(hsv, np.array([135, 35, 40]), np.array([175, 255, 255]))
    mask = cv2.bitwise_or(mask_blue, mask_violet)
    if int(mask.sum()) == 0:
        return image
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    out = image.copy()
    out[mask > 0] = (255, 255, 255)
    return out


def detect_skew_angle(gray: np.ndarray) -> float:
    """Оценка угла наклона текста в градусах (по горизонтальным линиям текста)."""
    if cv2 is None:
        return 0.0
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=gray.shape[1] // 4, maxLineGap=20)
    if lines is None:
        return 0.0
    angles: list[float] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 == x1:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) <= 15:  # игнорируем вертикальные линии и шум
            angles.append(angle)
    if not angles:
        return 0.0
    angle = float(np.median(angles))
    return angle if abs(angle) > 0.2 else 0.0


def deskew(image: np.ndarray) -> tuple[np.ndarray, float]:
    if cv2 is None:
        return image, 0.0
    gray = _to_gray(image)
    angle = detect_skew_angle(gray)
    if angle == 0.0:
        return image, 0.0
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, angle


def denoise_and_binarize(gray: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return gray
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # адаптивный порог устойчив к неравномерному освещению (фото со смартфона)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    return binary


def preprocess(image: np.ndarray, *, remove_seals: bool = True, binarize: bool = True) -> tuple[np.ndarray, dict]:
    """Полный конвейер препроцессинга. Возвращает изображение и метаданные шагов."""
    meta: dict = {"skew_angle": 0.0, "stamps_removed": False, "binarized": False}
    if cv2 is None:
        meta["engine"] = "opencv-not-installed"
        return image, meta

    working = image.copy()
    if remove_seals:
        before = working.copy()
        working = remove_stamps(working)
        meta["stamps_removed"] = bool(np.any(before != working))
    working, angle = deskew(working)
    meta["skew_angle"] = round(angle, 2)

    gray = _to_gray(working)
    if binarize:
        # бинаризация используется только как вход для OCR,
        # для отрисовки на фронтенде сохраняется исходная страница
        processed = denoise_and_binarize(gray)
        meta["binarized"] = True
        meta["engine"] = "opencv"
        return processed, meta
    meta["engine"] = "opencv"
    return gray, meta


def deskew_with_angle(image: np.ndarray, angle: float) -> np.ndarray:
    """Поворот на заданный угол (тот же угол применяется и к отображаемой странице, и к входу OCR)."""
    if cv2 is None or angle == 0.0:
        return image
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def prepare_page(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Подготовка страницы: возвращает (вход для OCR, изображение для показа, метаданные).

    Обе картинки имеют одинаковую геометрию, поэтому координаты слов из OCR
    совпадают с тем, что пользователь видит на скане (подсветка Bounding Boxes).
    На экран отдаётся цветная страница с печатями (их видно при проверке),
    а OCR получает версию без синих/фиолетовых печатей и после бинаризации.
    """
    meta: dict = {"skew_angle": 0.0, "stamps_removed": False, "binarized": False}
    if cv2 is None:
        meta["engine"] = "opencv-not-installed"
        return image, image, meta

    with stage("extract_pages/preprocess/stamps"):
        cleaned = remove_stamps(image)
        meta["stamps_removed"] = bool(np.any(cleaned != image))
    with stage("extract_pages/preprocess/skew_detect"):
        gray = _to_gray(cleaned)
        angle = detect_skew_angle(gray)
    meta["skew_angle"] = round(angle, 2)

    with stage("extract_pages/preprocess/deskew_display"):
        display = deskew_with_angle(image, angle)
    with stage("extract_pages/preprocess/deskew_gray"):
        gray_rot = deskew_with_angle(gray, angle)
    with stage("extract_pages/preprocess/denoise_binarize"):
        ocr_input = denoise_and_binarize(gray_rot)
    meta["binarized"] = True
    meta["engine"] = "opencv"
    return ocr_input, display, meta
