"""
Оптимізований препроцесор для рукописного тексту
handwrite2print/app/model/optimized_preprocessor.py
"""
import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OptimizedPreprocessor:
    """
    Оптимізований препроцесор для рукописного тексту
    Використовує найкращі практики для покращення якості OCR
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Ініціалізація препроцесора
        
        Args:
            config: словник конфігурації
        """
        self.config = config or {
            'binarization': False,  # Вимикаємо для рукописного тексту
            'denoise': True,
            'deskew': False,  # Вимикаємо для рукописного тексту
            'enhance_contrast': True,
            'sharpen': False,  # Вимикаємо для рукописного тексту
            'morphology': False,  # Вимикаємо для рукописного тексту
            'upscale_small': True,
            'remove_borders': False,
            'light_denoise': True  # Легке видалення шумів
        }
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Повна обробка зображення з валідацією
        
        Args:
            image: вхідне зображення (BGR, RGB або grayscale)
            
        Returns:
            оброблене зображення (grayscale, uint8)
            
        Raises:
            ValueError: якщо зображення некоректне або порожнє
        """
        try:
            # Валідація вхідного зображення
            if image is None:
                raise ValueError("Зображення не може бути None")
            if image.size == 0:
                raise ValueError("Зображення порожнє")
            if len(image.shape) < 2 or len(image.shape) > 3:
                raise ValueError(f"Невірна розмірність зображення: {image.shape}")
            
            # Конвертація в grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Перевірка типу даних
            if gray.dtype != np.uint8:
                gray = gray.astype(np.uint8)
            
            # 1. Збільшення маленьких зображень
            if self.config.get('upscale_small', True):
                gray = self._upscale_if_needed(gray)
            
            # 2. Видалення рамок (опціонально)
            if self.config.get('remove_borders', False):
                gray = self._remove_borders(gray)
            
            # 3. Покращення контрасту
            if self.config.get('enhance_contrast', True):
                gray = self._enhance_contrast(gray)
            
            # 4. Легке видалення шумів (для рукописного тексту)
            if self.config.get('light_denoise', True):
                gray = self._light_denoise(gray)
            elif self.config.get('denoise', False):
                gray = self._denoise(gray)
            
            # 5. Покращення різкості (тільки якщо увімкнено)
            if self.config.get('sharpen', False):
                gray = self._sharpen(gray)
            
            # 6. Бінаризація (тільки якщо увімкнено)
            if self.config.get('binarization', False):
                gray = self._binarize(gray)
            
            # 7. Морфологічні операції (тільки якщо увімкнено)
            if self.config.get('morphology', False):
                gray = self._morphological_operations(gray)
            
            # 8. Вирівнювання (deskewing)
            if self.config.get('deskew', True):
                gray = self._deskew(gray)
            
            # 9. Фінальна нормалізація та покращення
            gray = self._final_enhancement(gray)
            
            # Фінальна перевірка
            if gray.dtype != np.uint8:
                gray = gray.astype(np.uint8)
            
            return gray
        except Exception as e:
            logger.error(f"Помилка обробки зображення: {e}")
            # Повертаємо оригінальне зображення у випадку помилки
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image.astype(np.uint8) if image.dtype != np.uint8 else image
    
    def _upscale_if_needed(self, image: np.ndarray, min_size: int = 300) -> np.ndarray:
        """Збільшення маленьких зображень"""
        height, width = image.shape[:2]
        if height < min_size or width < min_size:
            scale = max(min_size / height, min_size / width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return image
    
    def _remove_borders(self, image: np.ndarray, border_size: int = 10) -> np.ndarray:
        """Видалення рамок"""
        h, w = image.shape[:2]
        if h > 2 * border_size and w > 2 * border_size:
            return image[border_size:h-border_size, border_size:w-border_size]
        return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Покращення контрасту через CLAHE"""
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        return enhanced
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Видалення шумів"""
        # Білатеральний фільтр зберігає краї
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        return denoised
    
    def _light_denoise(self, image: np.ndarray) -> np.ndarray:
        """Легке видалення шумів для рукописного тексту"""
        # Дуже легке гауссове згладжування
        denoised = cv2.GaussianBlur(image, (3, 3), 0)
        return denoised
    
    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """Покращення різкості"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        # Обмежуємо значення до 0-255
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """
        Розумна бінаризація
        Вибирає між адаптивною та Otsu залежно від якості
        """
        # Адаптивна бінаризація (краще для неоднорідного освітлення)
        binary_adaptive = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Otsu бінаризація
        _, binary_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Вибираємо кращий варіант
        adaptive_ratio = np.sum(binary_adaptive == 255) / binary_adaptive.size
        otsu_ratio = np.sum(binary_otsu == 255) / binary_otsu.size
        
        # Адаптивна краща, якщо має розумний баланс (20-80%)
        if 0.2 <= adaptive_ratio <= 0.8:
            return binary_adaptive
        else:
            return binary_otsu
    
    def _morphological_operations(self, image: np.ndarray) -> np.ndarray:
        """Морфологічні операції для покращення тексту"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        # Замикання - заповнює прогалини
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        # Відкриття - видаляє шуми
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        return opened
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Вирівнювання нахилу тексту
        """
        # Знаходимо координати ненульових пікселів
        coords = np.column_stack(np.where(image > 0))
        
        if len(coords) < 5:
            return image
        
        # Обчислюємо кут нахилу
        angle = cv2.minAreaRect(coords)[-1]
        
        # Корекція кута
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Обмеження кута (не більше 45 градусів)
        if abs(angle) > 45:
            return image
        
        # Поворот зображення
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def _final_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Фінальне покращення зображення для рукописного тексту
        
        Args:
            image: оброблене зображення
            
        Returns:
            фінально оброблене зображення
        """
        # Легке покращення контрасту (менш агресивне)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # Не застосовуємо морфологію та згладжування для рукописного тексту
        # Це може зіпсувати форму букв
        
        return enhanced
    
    def process_from_path(self, image_path: str) -> np.ndarray:
        """
        Обробка зображення з файлу
        
        Args:
            image_path: шлях до зображення
            
        Returns:
            оброблене зображення
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Не вдалося завантажити зображення: {image_path}")
            return self.process(image)
        except Exception as e:
            logger.error(f"Помилка завантаження зображення: {e}")
            raise

