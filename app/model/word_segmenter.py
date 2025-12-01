"""
Сегментація рукописного тексту на слова
handwrite2print/app/model/word_segmenter.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class WordSegment:
    """Розбитий сегмент слова"""
    bbox: Tuple[int, int, int, int]
    image: np.ndarray


class WordSegmenter:
    """Утиліта для розбиття рукописного тексту на слова"""

    @staticmethod
    def segment_words(image: np.ndarray) -> List[WordSegment]:
        """
        Повертає список сегментів зі словами, упорядкованих зліва направо.

        Args:
            image: оброблене (градації сірого) зображення
        """
        if image is None:
            return []

        original = image.copy()
        if len(original.shape) == 3:
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            gray = original

        # Нормалізація та легке згладжування
        gray = cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Адаптивна бінаризація + інверсія (текст = білий)
        thresh = cv2.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=21,
            C=10,
        )

        # Оцінюємо "тонкість" штрихів (середня висота компонентів)
        contours_preview, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        avg_height = 0
        if contours_preview:
            heights = [cv2.boundingRect(cnt)[3] for cnt in contours_preview]
            avg_height = sum(heights) / len(heights)
        thin_mode = avg_height < gray.shape[0] * 0.18  # ~18% висоти холста

        smoothed = thresh
        if thin_mode:
            logger.info("[WordSegmenter] Активовано режим згладжування для тонких штрихів")
            thin_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_DILATE, thin_kernel, iterations=1)
            smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_ERODE, thin_kernel, iterations=1)

        # Морфологічне закриття для з'єднання букв усередині слова
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                max(5, gray.shape[1] // 80),
                max(5, gray.shape[0] // 80),
            ),
        )
        closed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, close_kernel, iterations=1)

        # Горизонтальна дилатація для з'єднання букв у слово, але без злиття різних слів
        horiz_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                max(15, gray.shape[1] // 35),
                max(3, gray.shape[0] // 100),
            ),
        )
        dilated = cv2.dilate(closed, horiz_kernel, iterations=1)

        # Пошук компонентів
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        segments: List[WordSegment] = []
        min_area = (gray.shape[0] * gray.shape[1]) * 0.002  # фільтр шуму
        min_height = gray.shape[0] * 0.15  # ігноруємо дуже низькі контури
        padding = max(4, int(min(gray.shape[:2]) * 0.03))

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < min_area or h < min_height:
                continue

            x0 = max(0, x - padding)
            y0 = max(0, y - padding)
            x1 = min(gray.shape[1], x + w + padding)
            y1 = min(gray.shape[0], y + h + padding)

            crop = original[y0:y1, x0:x1]
            if crop.size == 0:
                continue

            # Масштабуємо кожен сегмент до більшої висоти для покращення OCR
            upscale_factor = 1.6
            resized = cv2.resize(
                crop,
                None,
                fx=upscale_factor,
                fy=upscale_factor,
                interpolation=cv2.INTER_CUBIC,
            )

            segments.append(WordSegment(bbox=(x0, y0, x1, y1), image=resized))

        # Сортуємо зліва направо
        segments.sort(key=lambda seg: seg.bbox[0])

        logger.info(f"[WordSegmenter] Знайдено {len(segments)} сегмент(ів)")
        return segments

