"""
Попередня обробка зображень
handwrite2print/app/model/preprocess.py
"""
import cv2
import numpy as np
from PIL import Image


class ImagePreprocessor:
    """Клас для попередньої обробки зображень"""
    
    def __init__(self):
        self.config = {
            'binarization': True,
            'denoise': True,
            'deskew': True,
            'resize': True
        }
        
    def process(self, image_path):
        """
        Повна обробка зображення з покращенням для кращої точності OCR
        
        Args:
            image_path: шлях до зображення
            
        Returns:
            оброблене зображення (numpy array)
        """
        # Завантаження зображення
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Не вдалося завантажити зображення: {image_path}")
            
        # Перетворення в градації сірого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Збільшуємо розмір для кращого розпізнавання (якщо зображення маленьке)
        height, width = gray.shape[:2]
        if height < 300 or width < 300:
            gray = self.upscale_image(gray, scale_factor=2)
        
        # Покращення контрасту ПЕРЕД іншими операціями (для кращої точності)
        gray = self.enhance_contrast(gray)
        
        # Видалення шумів
        if self.config['denoise']:
            gray = self.denoise(gray)
        
        # Покращення різкості
        gray = self.sharpen_image(gray)
            
        # Бінаризація
        if self.config['binarization']:
            gray = self.binarize(gray)
        
        # Морфологічні операції для покращення тексту
        gray = self.morphological_operations(gray)
            
        # Вирівнювання
        if self.config['deskew']:
            gray = self.deskew(gray)
            
        # Нормалізація розміру (опціонально)
        if self.config['resize']:
            gray = self.resize_image(gray)
            
        return gray
        
    def denoise(self, image):
        """
        Видалення шумів з зображення
        
        Args:
            image: вхідне зображення
            
        Returns:
            зображення без шумів
        """
        # Використання білатерального фільтру для збереження країв
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Альтернативно: Non-local Means Denoising
        # denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        
        return denoised
        
    def binarize(self, image):
        """
        Бінаризація зображення (перетворення в чорно-біле)
        
        Args:
            image: вхідне зображення в градаціях сірого
            
        Returns:
            бінаризоване зображення
        """
        # Спочатку пробуємо адаптивну бінаризацію (краще для рукопису з неоднорідним освітленням)
        binary_adaptive = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Також пробуємо Otsu's binarization
        _, binary_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Вибираємо той, який має кращий контраст (більше білих пікселів на темному фоні)
        # Для рукописного тексту адаптивна часто краща
        adaptive_contrast = np.sum(binary_adaptive == 255) / binary_adaptive.size
        
        # Якщо адаптивна дає розумний баланс (20-80% білих пікселів), використовуємо її
        if 0.2 <= adaptive_contrast <= 0.8:
            return binary_adaptive
        else:
            return binary_otsu
        
    def deskew(self, image):
        """
        Вирівнювання (виправлення нахилу) зображення
        
        Args:
            image: вхідне зображення
            
        Returns:
            вирівняне зображення
        """
        # Визначення кута нахилу
        coords = np.column_stack(np.nonzero(image > 0))
        
        if len(coords) < 5:
            return image
            
        angle = cv2.minAreaRect(coords)[-1]
        
        # Корекція кута
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        # Обмеження кута (щоб уникнути надмірних поворотів)
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
        
    def resize_image(self, image, max_width=2000, max_height=2000):
        """
        Зміна розміру зображення зі збереженням пропорцій
        
        Args:
            image: вхідне зображення
            max_width: максимальна ширина
            max_height: максимальна висота
            
        Returns:
            зображення зі зміненим розміром
        """
        height, width = image.shape[:2]
        
        # Перевірка, чи потрібно змінювати розмір
        if width <= max_width and height <= max_height:
            return image
            
        # Обчислення коефіцієнта масштабування
        scale = min(max_width / width, max_height / height)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized = cv2.resize(
            image, (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )
        
        return resized
        
    def enhance_contrast(self, image):
        """
        Покращення контрасту зображення для кращої точності OCR
        
        Args:
            image: вхідне зображення
            
        Returns:
            зображення з покращеним контрастом
        """
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Збільшуємо clipLimit для кращого контрасту (3.0 для кращого розпізнавання)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # Додаткова нормалізація для покращення якості
        # Нормалізуємо до 0-255, якщо потрібно
        if enhanced.max() > 255 or enhanced.min() < 0:
            enhanced = cv2.normalize(enhanced, enhanced, 0, 255, cv2.NORM_MINMAX)
        
        return enhanced
    
    def sharpen_image(self, image):
        """
        Покращення різкості зображення
        
        Args:
            image: вхідне зображення
            
        Returns:
            зображення з покращеною різкістю
        """
        # Ядро для покращення різкості
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened
    
    def morphological_operations(self, image):
        """
        Морфологічні операції для покращення тексту
        
        Args:
            image: вхідне зображення
            
        Returns:
            оброблене зображення
        """
        # Створюємо ядро для морфологічних операцій
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        # Замикання (closing) - заповнюємо невеликі прогалини в символах
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        # Відкриття (opening) - видаляємо невеликі шуми
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        return opened
    
    def upscale_image(self, image, scale_factor=2):
        """
        Збільшення розміру зображення для кращого розпізнавання
        
        Args:
            image: вхідне зображення
            scale_factor: коефіцієнт масштабування
            
        Returns:
            збільшене зображення
        """
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Використовуємо INTER_CUBIC для кращої якості при збільшенні
        upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return upscaled
        
    def remove_borders(self, image, border_size=10):
        """
        Видалення рамок з зображення
        
        Args:
            image: вхідне зображення
            border_size: розмір рамки для видалення
            
        Returns:
            зображення без рамок
        """
        h, w = image.shape[:2]
        
        # Видалення рамок
        cropped = image[border_size:h-border_size, border_size:w-border_size]
        
        return cropped
        
    def set_config(self, config):
        """
        Встановлення конфігурації обробки
        
        Args:
            config: словник з налаштуваннями
        """
        self.config.update(config)
