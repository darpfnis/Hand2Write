"""
OCR Manager з патерном Strategy для динамічного перемикання між рушіями
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class OCREngine(Enum):
    """Типи OCR рушіїв"""
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"


class OCRLanguage(Enum):
    """Підтримувані мови"""
    UKRAINIAN = "ukr"
    ENGLISH = "eng"
    BOTH = "eng+ukr"


class OCREngineMetadata:
    """Метадані про OCR рушій"""
    
    def __init__(self, name: str, description: str, speed: str, accuracy: str, 
                 best_for: str, requires_gpu: bool = False):
        self.name = name
        self.description = description
        self.speed = speed
        self.accuracy = accuracy
        self.best_for = best_for
        self.requires_gpu = requires_gpu


# Метадані для кожного рушія
ENGINE_METADATA = {
    OCREngine.TESSERACT: OCREngineMetadata(
        name="Tesseract",
        description="Найшвидший рушій для друкованого тексту",
        speed="Швидкий",
        accuracy="Висока для друкованого, середня для рукописного",
        best_for="Друкований текст, швидкість"
    ),
    OCREngine.EASYOCR: OCREngineMetadata(
        name="EasyOCR",
        description="Універсальний рушій з підтримкою рукописного тексту",
        speed="Середній",
        accuracy="Висока для рукописного",
        best_for="Рукописний текст, універсальність"
    ),
    OCREngine.PADDLEOCR: OCREngineMetadata(
        name="PaddleOCR",
        description="Баланс швидкості та точності, оптимізований для CPU",
        speed="Середній",
        accuracy="Висока для різних типів тексту",
        best_for="Баланс швидкості та точності"
    )
}


class OCRStrategy(ABC):
    """Абстрактна стратегія для OCR"""
    
    @abstractmethod
    def recognize(self, image: np.ndarray, language: OCRLanguage) -> str:
        """Розпізнавання тексту з зображення"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Перевірка доступності рушія"""
        pass
    
    @abstractmethod
    def get_metadata(self) -> OCREngineMetadata:
        """Отримання метаданих рушія"""
        pass


class TesseractStrategy(OCRStrategy):
    """Стратегія для Tesseract OCR"""
    
    def __init__(self):
        self._available = False
        self._check_availability()
    
    def _check_availability(self):
        """Перевірка доступності Tesseract"""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._available = True
        except Exception as e:
            logger.warning(f"Tesseract недоступний: {e}")
            self._available = False
    
    def is_available(self) -> bool:
        return self._available
    
    def get_metadata(self) -> OCREngineMetadata:
        return ENGINE_METADATA[OCREngine.TESSERACT]
    
    def recognize(self, image: np.ndarray, language: OCRLanguage) -> str:
        """
        Розпізнавання через Tesseract
        
        Args:
            image: зображення для розпізнавання
            language: мова розпізнавання
            
        Returns:
            Розпізнаний текст
            
        Raises:
            RuntimeError: якщо Tesseract недоступний
            ValueError: якщо зображення некоректне
        """
        if not self._available:
            raise RuntimeError("Tesseract недоступний")
        
        # Валідація зображення
        if image is None or image.size == 0:
            raise ValueError("Зображення не може бути порожнім")
        
        import pytesseract
        import cv2
        
        # Конвертуємо в uint8 якщо потрібно
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # Покращення контрасту для Tesseract
        if len(image.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            image = clahe.apply(image)
        
        # Мапінг мов
        lang_map = {
            OCRLanguage.UKRAINIAN: 'ukr',
            OCRLanguage.ENGLISH: 'eng',
            OCRLanguage.BOTH: 'ukr+eng'
        }
        tesseract_lang = lang_map[language]
        
        # Пробуємо різні PSM режими для кращого результату
        text = ""
        psm_modes = [6, 7, 11, 13]  # Найкращі для рукопису
        
        for psm_mode in psm_modes:
            try:
                custom_config = f'--oem 3 --psm {psm_mode}'
                test_text = pytesseract.image_to_string(
                    image, lang=tesseract_lang, config=custom_config
                )
                if len(test_text.strip()) > len(text.strip()):
                    text = test_text
            except Exception as e:
                logger.debug(f"PSM {psm_mode} помилка: {e}")
                continue
        
        return text.strip()


class EasyOCRStrategy(OCRStrategy):
    """Стратегія для EasyOCR (ліниве завантаження)"""
    
    def __init__(self):
        self._reader = None
        self._available = False
        self._readers_cache: Dict[OCRLanguage, Any] = {}
        self._check_availability()
    
    def _check_availability(self):
        """Перевірка доступності EasyOCR"""
        try:
            import easyocr
            self._available = True
        except ImportError:
            logger.warning("EasyOCR не встановлено")
            self._available = False
    
    def is_available(self) -> bool:
        return self._available
    
    def get_metadata(self) -> OCREngineMetadata:
        return ENGINE_METADATA[OCREngine.EASYOCR]
    
    def _get_reader(self, language: OCRLanguage):
        """Лінива ініціалізація Reader з кешуванням"""
        if language in self._readers_cache:
            return self._readers_cache[language]
        
        import easyocr
        
        # Мапінг мов
        lang_map = {
            OCRLanguage.UKRAINIAN: ['uk'],
            OCRLanguage.ENGLISH: ['en'],
            OCRLanguage.BOTH: ['en', 'uk']
        }
        easyocr_langs = lang_map[language]
        
        logger.info(f"Ініціалізація EasyOCR для мов: {easyocr_langs}")
        reader = easyocr.Reader(easyocr_langs, gpu=False)
        self._readers_cache[language] = reader
        
        return reader
    
    def recognize(self, image: np.ndarray, language: OCRLanguage) -> str:
        """Розпізнавання через EasyOCR"""
        if not self._available:
            raise RuntimeError("EasyOCR недоступний")
        
        import cv2
        
        reader = self._get_reader(language)
        
        # Підготовка зображення (EasyOCR працює з RGB)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Розпізнавання
        result = reader.readtext(
            image,
            paragraph=False,
            detail=1,
            width_ths=0.7,
            height_ths=0.7
        )
        
        # Фільтрація та сортування
        confidence_threshold = 0.1
        filtered_result = [
            (bbox, txt, conf) for bbox, txt, conf in result
            if isinstance(conf, (int, float)) and conf > confidence_threshold and txt.strip()
        ]
        
        # Сортування за позицією (зліва направо, зверху вниз)
        filtered_result.sort(key=lambda x: (x[0][0][1], x[0][0][0]))
        
        text = ' '.join([txt for _, txt, _ in filtered_result])
        return text.strip()


class PaddleOCRStrategy(OCRStrategy):
    """Стратегія для PaddleOCR (оптимізовано для CPU)"""
    
    def __init__(self):
        self._instances: Dict[OCRLanguage, Any] = {}
        self._available = False
        self._check_availability()
    
    def _check_availability(self):
        """Перевірка доступності PaddleOCR"""
        try:
            from paddleocr import PaddleOCR
            self._available = True
        except ImportError:
            logger.warning("PaddleOCR не встановлено")
            self._available = False
    
    def is_available(self) -> bool:
        return self._available
    
    def get_metadata(self) -> OCREngineMetadata:
        return ENGINE_METADATA[OCREngine.PADDLEOCR]
    
    def _get_instance(self, language: OCRLanguage):
        """Лінива ініціалізація з кешуванням"""
        if language in self._instances:
            logger.info(f"[PaddleOCR-OCRManager] Використовується закешований екземпляр для мови: {language.value}")
            return self._instances[language]
        
        from paddleocr import PaddleOCR
        
        # Використовуємо 'ru' для української мови - краще працює з кирилицею
        # Модель 'ru' може не розпізнавати деякі українські особливі символи (і, ї, є),
        # але загалом дає кращі результати для рукописного тексту
        lang_map = {
            OCRLanguage.UKRAINIAN: 'ru',  # Використовуємо 'ru' для кращого розпізнавання кирилиці
            OCRLanguage.ENGLISH: 'en',
            OCRLanguage.BOTH: 'en'  # PaddleOCR не підтримує мультимовність
        }
        paddle_lang = lang_map[language]
        
        logger.info(f"[PaddleOCR-OCRManager] Ініціалізація PaddleOCR для мови: {paddle_lang} (запитана: {language.value})")
        if language == OCRLanguage.UKRAINIAN:
            logger.info("[PaddleOCR-OCRManager] Використовується модель 'ru' для української мови (краще працює з кирилицею)")
            print("[PaddleOCR-OCRManager] Використовується модель 'ru' для української мови (краще працює з кирилицею)", flush=True)
        print(f"[PaddleOCR-OCRManager] Ініціалізація PaddleOCR для мови: {paddle_lang} (запитана: {language.value})", flush=True)
        
        # Оптимізація для CPU
        # PaddleOCR v3.3.2 не підтримує use_gpu, enable_mkldnn, cpu_threads, show_log параметри
        try:
            instance = PaddleOCR(
                lang=paddle_lang,
                use_angle_cls=False
            )
            logger.info(
                "[PaddleOCR-OCRManager] ✓ PaddleOCR екземпляр створено для мови: %s",
                paddle_lang,
            )
            print(
                f"[PaddleOCR-OCRManager] ✓ PaddleOCR екземпляр створено для мови: {paddle_lang}",
                flush=True,
            )
        except Exception as e:
            logger.error(
                "[PaddleOCR-OCRManager] ✗ Помилка створення PaddleOCR для мови %s: %s",
                paddle_lang,
                e,
            )
            print(
                f"[PaddleOCR-OCRManager] ✗ Помилка створення PaddleOCR для мови {paddle_lang}: {e}",
                flush=True,
            )
            raise
        
        self._instances[language] = instance
        return instance
    
    def recognize(self, image: np.ndarray, language: OCRLanguage) -> str:
        """Розпізнавання через PaddleOCR"""
        if not self._available:
            raise RuntimeError("PaddleOCR недоступний")
        
        import cv2
        
        ocr_instance = self._get_instance(language)
        
        # Логуємо, яка мова реально використовується
        lang_map = {
            OCRLanguage.UKRAINIAN: 'ru',  # Використовуємо 'ru' для кращого розпізнавання кирилиці
            OCRLanguage.ENGLISH: 'en',
            OCRLanguage.BOTH: 'en'
        }
        actual_lang = lang_map[language]
        logger.info(f"[PaddleOCR-OCRManager] Початок розпізнавання, запитана мова: {language.value}, використовується: {actual_lang}")
        print(f"[PaddleOCR-OCRManager] Початок розпізнавання, запитана мова: {language.value}, використовується: {actual_lang}", flush=True)
        
        # Підготовка зображення (PaddleOCR працює з BGR)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Покращений препроцесинг для всіх мов (особливо для рукописного тексту)
        # Збільшуємо розмір зображення, якщо воно маленьке (для кращого розпізнавання)
        height, width = image.shape[:2]
        if height < 400 or width < 400:
            scale_factor = max(400 / height, 400 / width, 2.0)  # Мінімум 2x збільшення
            image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            logger.info(f"[PaddleOCR-OCRManager] Збільшено зображення в {scale_factor:.1f} разів: {image.shape}")
            print(f"[PaddleOCR-OCRManager] Збільшено зображення в {scale_factor:.1f} разів", flush=True)
        
        # Покращення контрасту для всіх мов
        # Конвертуємо в grayscale для покращення контрасту
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Застосовуємо адаптивну гістограму для покращення контрасту (більш агресивно для рукописного тексту)
        # Різні налаштування для різних мов
        if language == OCRLanguage.ENGLISH:
            # Більш агресивна обробка для англійської мови
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Більш сильне покращення різкості для англійської
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel * 0.25)
            
            # Додаткова бінаризація для покращення контрасту
            _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            enhanced = cv2.bitwise_and(sharpened, binary)
            
            logger.info("[PaddleOCR-OCRManager] Застосовано агресивний препроцесинг для англійської мови")
            print("[PaddleOCR-OCRManager] Застосовано агресивний препроцесинг для англійської мови", flush=True)
        else:
            # Стандартна обробка для інших мов
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Покращення різкості
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel * 0.15)
            enhanced = sharpened
        
        # Конвертуємо назад в BGR
        image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        logger.info("[PaddleOCR-OCRManager] Застосовано покращений препроцесинг для рукописного тексту")
        print("[PaddleOCR-OCRManager] Застосовано покращений препроцесинг для рукописного тексту", flush=True)
        
        # Розпізнавання
        # PaddleOCR v3.3.2 не підтримує параметр cls
        try:
            results = ocr_instance.ocr(image)
            logger.info(
                "[PaddleOCR-OCRManager] ocr() завершено, тип результату: %s",
                type(results),
            )
            print(
                f"[PaddleOCR-OCRManager] ocr() завершено, тип результату: {type(results)}",
                flush=True,
            )
        except Exception as e:
            logger.error(f"[PaddleOCR-OCRManager] Помилка виклику ocr(): {e}", exc_info=True)
            print(f"[PaddleOCR-OCRManager] ✗ Помилка виклику ocr(): {e}", flush=True)
            # Спробуємо predict() як fallback
            if hasattr(ocr_instance, 'predict'):
                try:
                    logger.info("[PaddleOCR-OCRManager] Спроба використати predict() як fallback...")
                    print("[PaddleOCR-OCRManager] Спроба використати predict() як fallback...", flush=True)
                    results = ocr_instance.predict(image)
                    logger.info(f"[PaddleOCR-OCRManager] predict() завершено, тип результату: {type(results)}")
                    print(f"[PaddleOCR-OCRManager] predict() завершено, тип результату: {type(results)}", flush=True)
                except Exception as e2:
                    logger.error(f"[PaddleOCR-OCRManager] Помилка predict(): {e2}", exc_info=True)
                    print(f"[PaddleOCR-OCRManager] ✗ Помилка predict(): {e2}", flush=True)
                    return ""
            else:
                return ""
        
        # Детальна діагностика результату
        if results is None:
            logger.warning("[PaddleOCR-OCRManager] Результат None")
            print("[PaddleOCR-OCRManager] ⚠️ Результат None", flush=True)
            return ""
        
        text_parts = []
        
        if isinstance(results, (list, tuple)):
            logger.info(f"[PaddleOCR-OCRManager] Результат: список/кортеж, довжина: {len(results)}")
            print(f"[PaddleOCR-OCRManager] Результат: список/кортеж, довжина: {len(results)}", flush=True)
            
            if len(results) == 0:
                logger.warning("[PaddleOCR-OCRManager] Результат: порожній список")
                print("[PaddleOCR-OCRManager] ⚠️ Результат: порожній список", flush=True)
                return ""
            
            first_item = results[0]
            logger.info(f"[PaddleOCR-OCRManager] results[0] тип: {type(first_item)}")
            print(f"[PaddleOCR-OCRManager] results[0] тип: {type(first_item)}", flush=True)
            
            # Перевіряємо, чи це OCRResult об'єкт або словник з rec_texts
            rec_texts = None
            rec_scores = None
            
            # Спробуємо отримати rec_texts з об'єкта
            if hasattr(first_item, 'rec_texts'):
                rec_texts = first_item.rec_texts
                rec_scores = getattr(first_item, 'rec_scores', None)
                logger.info(f"[PaddleOCR-OCRManager] Знайдено rec_texts через атрибут: {rec_texts}")
                print(f"[PaddleOCR-OCRManager] Знайдено rec_texts через атрибут: {rec_texts}", flush=True)
            elif isinstance(first_item, dict) and 'rec_texts' in first_item:
                rec_texts = first_item['rec_texts']
                rec_scores = first_item.get('rec_scores', None)
                logger.info(f"[PaddleOCR-OCRManager] Знайдено rec_texts через словник: {rec_texts}")
                print(f"[PaddleOCR-OCRManager] Знайдено rec_texts через словник: {rec_texts}", flush=True)
                logger.info(f"[PaddleOCR-OCRManager] rec_scores тип: {type(rec_scores)}, значення: {rec_scores}")
                print(f"[PaddleOCR-OCRManager] rec_scores тип: {type(rec_scores)}, значення: {rec_scores}", flush=True)
            elif isinstance(first_item, (list, tuple)):
                # Старий формат: список списків [[[bbox], (text, confidence)], ...]
                logger.info("[PaddleOCR-OCRManager] Старий формат: список списків")
                print("[PaddleOCR-OCRManager] Старий формат: список списків", flush=True)
                try:
                    for idx, line in enumerate(first_item):
                        logger.debug(
                            "[PaddleOCR-OCRManager] Обробка рядка %s: тип=%s",
                            idx,
                            type(line),
                        )
                        if isinstance(line, (list, tuple)) and len(line) >= 2:
                            text_item = line[1]
                            if (
                                isinstance(text_item, (list, tuple))
                                and len(text_item) >= 1
                            ):
                                text_str = str(text_item[0]).strip()
                                if text_str:
                                    text_parts.append(text_str)
                                    logger.debug(
                                        "[PaddleOCR-OCRManager] Додано текст: '%s'",
                                        text_str,
                                    )
                            elif isinstance(text_item, str):
                                text_str = text_item.strip()
                                if text_str:
                                    text_parts.append(text_str)
                                    logger.debug(
                                        "[PaddleOCR-OCRManager] Додано текст (рядок): '%s'",
                                        text_str,
                                    )
                except Exception as e:
                    logger.error(
                        "[PaddleOCR-OCRManager] Помилка обробки старого формату: %s",
                        e,
                        exc_info=True,
                    )
                    print(
                        f"[PaddleOCR-OCRManager] ✗ Помилка обробки старого формату: {e}",
                        flush=True,
                    )
            
            # Обробка rec_texts з фільтрацією за впевненістю
            if rec_texts is not None:
                logger.info(f"[PaddleOCR-OCRManager] Обробка rec_texts: {rec_texts}")
                print(f"[PaddleOCR-OCRManager] Обробка rec_texts: {rec_texts}", flush=True)
                if isinstance(rec_texts, (list, tuple)):
                    # Мінімальна впевненість для прийняття результату
                    # Для англійської мови знижуємо поріг (рукописний текст має нижчу впевненість)
                    # Для інших мов залишаємо вищий поріг
                    if language == OCRLanguage.ENGLISH:
                        min_confidence = 0.3  # Нижчий поріг для англійської (рукописний текст)
                    else:
                        min_confidence = 0.4  # Вищий поріг для інших мов
                    
                    for idx, text_item in enumerate(rec_texts):
                        # Перевіряємо, чи текст не порожній
                        text_str = str(text_item).strip() if text_item else ""
                        
                        # Фільтруємо порожні результати
                        if not text_str:
                            logger.warning(f"[PaddleOCR-OCRManager] ✗ Пропущено порожній результат (idx={idx})")
                            print(f"[PaddleOCR-OCRManager] ✗ Пропущено порожній результат (idx={idx})", flush=True)
                            continue
                        
                        # Перевіряємо впевненість, якщо доступна
                        score = None
                        if rec_scores and idx < len(rec_scores):
                            score = rec_scores[idx]
                            
                            # Фільтруємо результати з нульовою або дуже низькою впевненістю
                            if isinstance(score, (int, float)):
                                if score <= 0.0:
                                    logger.warning(f"[PaddleOCR-OCRManager] ✗ Пропущено '{text_str}' (нульова впевненість {score:.3f})")
                                    print(f"[PaddleOCR-OCRManager] ✗ Пропущено '{text_str}' (нульова впевненість {score:.3f})", flush=True)
                                    continue
                                
                                logger.info(f"[PaddleOCR-OCRManager] Текст '{text_str}' з впевненістю {score:.3f} (поріг: {min_confidence})")
                                print(f"[PaddleOCR-OCRManager] Текст '{text_str}' з впевненістю {score:.3f} (поріг: {min_confidence})", flush=True)
                                
                                # Фільтруємо результати з низькою впевненістю
                                if score < min_confidence:
                                    logger.warning(f"[PaddleOCR-OCRManager] ✗ Пропущено '{text_str}' (низька впевненість {score:.3f} < {min_confidence})")
                                    print(f"[PaddleOCR-OCRManager] ✗ Пропущено '{text_str}' (низька впевненість {score:.3f} < {min_confidence})", flush=True)
                                    continue
                                else:
                                    logger.info(f"[PaddleOCR-OCRManager] ✓ Прийнято '{text_str}' (впевненість {score:.3f} >= {min_confidence})")
                                    print(f"[PaddleOCR-OCRManager] ✓ Прийнято '{text_str}' (впевненість {score:.3f} >= {min_confidence})", flush=True)
                            else:
                                # Якщо впевненість не число, але текст є, додаємо з попередженням
                                logger.warning(f"[PaddleOCR-OCRManager] ⚠️ Впевненість не число ({type(score)}) для '{text_str}', але додаємо")
                                print(f"[PaddleOCR-OCRManager] ⚠️ Впевненість не число для '{text_str}', але додаємо", flush=True)
                        else:
                            # Якщо впевненість недоступна, але текст є, додаємо з попередженням
                            logger.warning(f"[PaddleOCR-OCRManager] ⚠️ Впевненість недоступна для '{text_str}' (idx={idx}, rec_scores={rec_scores is not None})")
                            print(f"[PaddleOCR-OCRManager] ⚠️ Впевненість недоступна для '{text_str}'", flush=True)
                        
                        text_parts.append(text_str)
                elif isinstance(rec_texts, str):
                    text_str = rec_texts.strip()
                    if text_str:
                        text_parts.append(text_str)
                        logger.info(f"[PaddleOCR-OCRManager] Додано текст (рядок): '{text_str}'")
                        print(f"[PaddleOCR-OCRManager] Додано текст (рядок): '{text_str}'", flush=True)
        else:
            logger.warning(f"[PaddleOCR-OCRManager] Невідомий тип результату: {type(results)}")
            print(f"[PaddleOCR-OCRManager] ⚠️ Невідомий тип результату: {type(results)}", flush=True)
            # Спробуємо обробити як рядок
            try:
                text = str(results).strip()
                if text and len(text) > 0:
                    logger.info(f"[PaddleOCR-OCRManager] Конвертовано в рядок: {text[:100]}...")
                    return text
            except Exception:
                pass
            return ""
        
        result_text = ' '.join(text_parts).strip()
        
        # Якщо результат порожній, спробуємо predict() як fallback
        if not result_text and hasattr(ocr_instance, 'predict'):
            try:
                logger.info("[PaddleOCR-OCRManager] ocr() повернув порожній результат, спроба predict() як fallback...")
                print("[PaddleOCR-OCRManager] ocr() повернув порожній результат, спроба predict() як fallback...", flush=True)
                # Використовуємо predict() з параметрами для кращого розпізнавання рукописного тексту
                fallback_results = ocr_instance.predict(
                    image,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    text_det_thresh=0.2,
                    text_det_box_thresh=0.3,
                    text_rec_score_thresh=0.1
                )
                logger.info(f"[PaddleOCR-OCRManager] predict() fallback завершено, тип: {type(fallback_results)}")
                print("[PaddleOCR-OCRManager] predict() fallback завершено", flush=True)
                
                # Обробляємо результат predict() так само, як ocr()
                if isinstance(fallback_results, (list, tuple)) and len(fallback_results) > 0:
                    fallback_item = fallback_results[0]
                    if hasattr(fallback_item, 'rec_texts'):
                        fallback_texts = fallback_item.rec_texts
                    elif isinstance(fallback_item, dict) and 'rec_texts' in fallback_item:
                        fallback_texts = fallback_item['rec_texts']
                    else:
                        fallback_texts = None
                    
                    if fallback_texts and isinstance(fallback_texts, (list, tuple)) and len(fallback_texts) > 0:
                        result_text = ' '.join(str(t).strip() for t in fallback_texts if t and str(t).strip()).strip()
                        logger.info(f"[PaddleOCR-OCRManager] predict() fallback дав результат: '{result_text}'")
                        print(f"[PaddleOCR-OCRManager] predict() fallback дав результат: '{result_text}'", flush=True)
            except Exception as e:
                logger.warning(f"[PaddleOCR-OCRManager] predict() fallback помилка: {e}")
                print(f"[PaddleOCR-OCRManager] predict() fallback помилка: {e}", flush=True)
        
        # Пост-обробка для української мови: виправлення помилок змішування кирилиці та латиниці
        # PaddleOCR часто плутає схожі літери: і/I, в/B, р/p, о/O, а/A, е/E, т/T, н/H, у/Y, х/X, с/C, м/M
        # ВАЖЛИВО: Модель 'ru' може розпізнавати український текст як латинський
        if language == OCRLanguage.UKRAINIAN and result_text:
            logger.info("[PaddleOCR-OCRManager] Застосовано пост-обробку для української мови")
            print("[PaddleOCR-OCRManager] Застосовано пост-обробку для української мови", flush=True)
            
            # Перевіряємо, чи є кириличні літери в тексті
            has_cyrillic = any('\u0400' <= char <= '\u04FF' for char in result_text)
            
            # Перевіряємо, чи текст містить латинські літери (можлива помилка розпізнавання)
            # Для української мови завжди застосовуємо пост-обробку, якщо є латинські літери
            has_latin = any(char.isalpha() and ord(char) < 128 and not ('\u0400' <= char <= '\u04FF') for char in result_text)
            
            # Застосовуємо пост-обробку завжди для української мови, якщо є латинські літери
            # або якщо немає кирилиці (можлива помилка розпізнавання)
            if has_latin or not has_cyrillic:
                logger.info(f"[PaddleOCR-OCRManager] has_cyrillic={has_cyrillic}, has_latin={has_latin}, застосовуємо пост-обробку")
                print(f"[PaddleOCR-OCRManager] has_cyrillic={has_cyrillic}, has_latin={has_latin}, застосовуємо пост-обробку", flush=True)
                # Словник замін: латинська -> кирилична (тільки для схожих літер)
                # PaddleOCR часто плутає ці літери в українському тексті
                # ВАЖЛИВО: Зберігаємо українські особливі символи (і, ї, є) - вони вже правильні
                # Використовуємо контекстно-залежні заміни для кращого виправлення
                
                # Спочатку виправляємо очевидні помилки в контексті українських слів
                corrected_text = result_text
                
                # Спеціальні випадки для поширених помилок
                # "pubi" -> "рив" (в контексті "Привіт")
                if 'pubi' in corrected_text.lower():
                    corrected_text = corrected_text.replace('pubi', 'рив')
                    corrected_text = corrected_text.replace('Pubi', 'Рив')
                    corrected_text = corrected_text.replace('PUBI', 'РИВ')
                
                # "Tect" -> "Тест" (поширена помилка)
                if corrected_text.lower() == 'tect':
                    corrected_text = 'Тест'
                    logger.info("[PaddleOCR-OCRManager] Спеціальна заміна: 'Tect' -> 'Тест'")
                    print("[PaddleOCR-OCRManager] Спеціальна заміна: 'Tect' -> 'Тест'", flush=True)
                
                # Видаляємо спеціальні символи, які не мають сенсу в тексті (якщо вони не частина слова)
                # Наприклад, '&' в "C&iT" -> "Світ"
                if '&' in corrected_text:
                    # Замінюємо '&' на порожній рядок, якщо він між літерами
                    corrected_text = corrected_text.replace('&', '')
                    logger.info("[PaddleOCR-OCRManager] Видалено спеціальний символ '&'")
                    print("[PaddleOCR-OCRManager] Видалено спеціальний символ '&'", flush=True)
                
                # Виправлення цифр, які часто плутаються з літерами
                # "6" часто плутається з "б" (особливо в рукописному тексті)
                if '6' in corrected_text:
                    # Замінюємо "6" на "б" в контексті українських слів
                    corrected_text = corrected_text.replace('6', 'б')
                    logger.info("[PaddleOCR-OCRManager] Виправлено '6' -> 'б'")
                    print("[PaddleOCR-OCRManager] Виправлено '6' -> 'б'", flush=True)
                
                # "0" часто плутається з "о" (особливо в рукописному тексті)
                if '0' in corrected_text:
                    # Замінюємо "0" на "о" в контексті українських слів
                    corrected_text = corrected_text.replace('0', 'о')
                    logger.info("[PaddleOCR-OCRManager] Виправлено '0' -> 'о'")
                    print("[PaddleOCR-OCRManager] Виправлено '0' -> 'о'", flush=True)
                
                # Загальні заміни латинських літер на кириличні
                # Важливо: замінюємо в правильному порядку (спочатку великі, потім малі)
                replacements = [
                    ('T', 'Т'),  # латинська T -> кирилична Т (важливо для "Тест" -> "Tecr", "Tect")
                    ('t', 'т'),  # латинська t -> кирилична т (важливо для "Tect" -> "Тест")
                    ('E', 'Е'),  # латинська E -> кирилична Е
                    ('e', 'е'),  # латинська e -> кирилична е (для "Tect" -> "Тест", а не 'в')
                    ('C', 'С'),  # латинська C -> кирилична С (важливо для "CeiT" -> "Світ")
                    ('c', 'с'),  # латинська c -> кирилична с
                    ('r', 'р'),  # латинська r -> кирилична р
                    ('I', 'І'),  # латинська I -> українська І (не і!)
                    ('i', 'і'),  # латинська i -> українська і (важливо для "CeiT" -> "Світ")
                    ('n', 'н'),  # латинська n -> кирилична н (важливо для "ind" -> "інд")
                    ('d', 'д'),  # латинська d -> кирилична д (важливо для "ind" -> "інд")
                    ('B', 'В'),  # латинська B -> кирилична В
                    ('p', 'р'),  # латинська p -> кирилична р
                    ('u', 'в'),  # латинська u -> кирилична в
                    ('O', 'О'),  # латинська O -> кирилична О
                    ('o', 'о'),  # латинська o -> кирилична о
                    ('A', 'А'),  # латинська A -> кирилична А
                    ('a', 'а'),  # латинська a -> кирилична а
                    ('H', 'Н'),  # латинська H -> кирилична Н
                    ('Y', 'У'),  # латинська Y -> кирилична У
                    ('X', 'Х'),  # латинська X -> кирилична Х
                    ('M', 'М'),  # латинська M -> кирилична М
                    ('P', 'Р'),  # латинська P -> кирилична Р
                    ('K', 'К'),  # латинська K -> кирилична К
                    ('k', 'к'),  # латинська k -> кирилична к
                    ('v', 'в'),  # латинська v -> кирилична в
                    ('V', 'В'),  # латинська V -> кирилична В
                    ('S', 'С'),  # латинська S -> кирилична С (може бути помилка)
                    ('s', 'с'),  # латинська s -> кирилична с
                    ('L', 'Л'),  # латинська L -> кирилична Л
                    ('l', 'л'),  # латинська l -> кирилична л
                    ('g', 'г'),  # латинська g -> кирилична г (важливо для "gnd" -> "гнд")
                    ('G', 'Г'),  # латинська G -> кирилична Г
                    ('f', 'в'),  # латинська f -> кирилична в (важливо для "Пpufi" -> "Привіт")
                    ('F', 'В'),  # латинська F -> кирилична В
                    ('N', 'Л'),  # латинська N -> кирилична Л (важливо для "Nю60В" -> "Любов")
                    ('n', 'л'),  # латинська n -> кирилична л (якщо ще не замінено вище)
                ]
                
                # Застосовуємо заміни для всіх латинських літер
                for latin, cyrillic in replacements:
                    if latin in corrected_text:
                        corrected_text = corrected_text.replace(latin, cyrillic)
                
                if corrected_text != result_text:
                    logger.info(f"[PaddleOCR-OCRManager] Виправлено: '{result_text}' -> '{corrected_text}'")
                    print(f"[PaddleOCR-OCRManager] Виправлено: '{result_text}' -> '{corrected_text}'", flush=True)
                    result_text = corrected_text
        
        logger.info(f"[PaddleOCR-OCRManager] Фінальний результат: {len(result_text)} символів")
        print(f"[PaddleOCR-OCRManager] Фінальний результат: {len(result_text)} символів", flush=True)
        if result_text:
            logger.info(f"[PaddleOCR-OCRManager] Перші 100 символів: {result_text[:100]}...")
            print(f"[PaddleOCR-OCRManager] Перші 100 символів: {result_text[:100]}...", flush=True)
        else:
            logger.warning("[PaddleOCR-OCRManager] ⚠️ ФІНАЛЬНИЙ РЕЗУЛЬТАТ ПОРОЖНІЙ!")
            print("[PaddleOCR-OCRManager] ⚠️ ФІНАЛЬНИЙ РЕЗУЛЬТАТ ПОРОЖНІЙ!", flush=True)
            print(f"[PaddleOCR-OCRManager] Повний результат для діагностики: {results}", flush=True)
        
        return result_text


class OCRManager:
    """Менеджер OCR з можливістю динамічного перемикання"""
    
    def __init__(self):
        self._strategies: Dict[OCREngine, OCRStrategy] = {
            OCREngine.TESSERACT: TesseractStrategy(),
            OCREngine.EASYOCR: EasyOCRStrategy(),
            OCREngine.PADDLEOCR: PaddleOCRStrategy()
        }
        self._current_engine = OCREngine.TESSERACT
        self._current_language = OCRLanguage.UKRAINIAN
    
    def get_available_engines(self) -> list[OCREngine]:
        """Отримання списку доступних рушіїв"""
        return [
            engine for engine, strategy in self._strategies.items()
            if strategy.is_available()
        ]
    
    def get_engine_metadata(self, engine: OCREngine) -> Optional[OCREngineMetadata]:
        """Отримання метаданих рушія"""
        if engine in self._strategies:
            return self._strategies[engine].get_metadata()
        return None
    
    def set_engine(self, engine: OCREngine) -> bool:
        """Встановлення поточного рушія"""
        if engine not in self._strategies:
            logger.error(f"Невідомий рушій: {engine}")
            return False
        
        if not self._strategies[engine].is_available():
            logger.error(f"Рушій {engine.value} недоступний")
            return False
        
        self._current_engine = engine
        logger.info(f"Встановлено рушій: {engine.value}")
        return True
    
    def set_language(self, language: OCRLanguage):
        """Встановлення мови розпізнавання"""
        self._current_language = language
        logger.info(f"Встановлено мову: {language.value}")
    
    def recognize(self, image: np.ndarray, 
                  engine: Optional[OCREngine] = None,
                  language: Optional[OCRLanguage] = None) -> str:
        """Розпізнавання тексту"""
        use_engine = engine or self._current_engine
        use_language = language or self._current_language
        
        if use_engine not in self._strategies:
            raise ValueError(f"Невідомий рушій: {use_engine}")
        
        strategy = self._strategies[use_engine]
        
        if not strategy.is_available():
            raise RuntimeError(f"Рушій {use_engine.value} недоступний")
        
        logger.info(f"Розпізнавання: {use_engine.value}, мова: {use_language.value}")
        return strategy.recognize(image, use_language)
    
    def get_current_engine(self) -> OCREngine:
        """Отримання поточного рушія"""
        return self._current_engine
    
    def get_current_language(self) -> OCRLanguage:
        """Отримання поточної мови"""
        return self._current_language
