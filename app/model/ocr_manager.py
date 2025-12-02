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
        actual_lang = self._get_actual_language(language)
        self._log_recognition_start(language, actual_lang)
        
        processed_image = self._prepare_image_for_paddleocr(image, language)
        results = self._perform_ocr_recognition(ocr_instance, processed_image)
        
        if results is None:
            return ""
        
        text_parts = self._parse_ocr_results(results, language)
        result_text = ' '.join(text_parts).strip()
        
        if not result_text:
            result_text = self._try_predict_fallback(ocr_instance, processed_image)
        
        if language == OCRLanguage.UKRAINIAN and result_text:
            result_text = self._apply_ukrainian_postprocessing(result_text)
        
        self._log_final_result(result_text, results)
        return result_text
    
    def _get_actual_language(self, language: OCRLanguage) -> str:
        """Отримання фактичної мови для PaddleOCR"""
        lang_map = {
            OCRLanguage.UKRAINIAN: 'ru',
            OCRLanguage.ENGLISH: 'en',
            OCRLanguage.BOTH: 'en'
        }
        return lang_map[language]
    
    def _log_recognition_start(self, language: OCRLanguage, actual_lang: str) -> None:
        """Логування початку розпізнавання"""
        logger.info(f"[PaddleOCR-OCRManager] Початок розпізнавання, запитана мова: {language.value}, використовується: {actual_lang}")
        print(f"[PaddleOCR-OCRManager] Початок розпізнавання, запитана мова: {language.value}, використовується: {actual_lang}", flush=True)
    
    def _prepare_image_for_paddleocr(self, image: np.ndarray, language: OCRLanguage) -> np.ndarray:
        """Підготовка зображення для PaddleOCR"""
        import cv2
        
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        image = self._resize_image_if_needed(image)
        image = self._enhance_image_contrast(image, language)
        
        logger.info("[PaddleOCR-OCRManager] Застосовано покращений препроцесинг для рукописного тексту")
        print("[PaddleOCR-OCRManager] Застосовано покращений препроцесинг для рукописного тексту", flush=True)
        return image
    
    def _resize_image_if_needed(self, image: np.ndarray) -> np.ndarray:
        """Збільшення зображення, якщо воно занадто маленьке"""
        import cv2
        height, width = image.shape[:2]
        
        if height < 400 or width < 400:
            scale_factor = max(400 / height, 400 / width, 2.0)
            image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            logger.info(f"[PaddleOCR-OCRManager] Збільшено зображення в {scale_factor:.1f} разів: {image.shape}")
            print(f"[PaddleOCR-OCRManager] Збільшено зображення в {scale_factor:.1f} разів", flush=True)
        
        return image
    
    def _enhance_image_contrast(self, image: np.ndarray, language: OCRLanguage) -> np.ndarray:
        """Покращення контрасту зображення"""
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if language == OCRLanguage.ENGLISH:
            enhanced = self._apply_aggressive_enhancement(gray)
        else:
            enhanced = self._apply_standard_enhancement(gray)
        
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    def _apply_aggressive_enhancement(self, gray: np.ndarray) -> np.ndarray:
        """Агресивне покращення для англійської мови"""
        import cv2
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel * 0.25)
        
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        enhanced = cv2.bitwise_and(sharpened, binary)
        
        logger.info("[PaddleOCR-OCRManager] Застосовано агресивний препроцесинг для англійської мови")
        print("[PaddleOCR-OCRManager] Застосовано агресивний препроцесинг для англійської мови", flush=True)
        return enhanced
    
    def _apply_standard_enhancement(self, gray: np.ndarray) -> np.ndarray:
        """Стандартне покращення для інших мов"""
        import cv2
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel * 0.15)
        return sharpened
    
    def _perform_ocr_recognition(self, ocr_instance, image: np.ndarray):
        """Виконання OCR розпізнавання"""
        try:
            results = ocr_instance.ocr(image)
            logger.info("[PaddleOCR-OCRManager] ocr() завершено, тип результату: %s", type(results))
            print(f"[PaddleOCR-OCRManager] ocr() завершено, тип результату: {type(results)}", flush=True)
            return results
        except Exception as e:
            logger.error(f"[PaddleOCR-OCRManager] Помилка виклику ocr(): {e}", exc_info=True)
            print(f"[PaddleOCR-OCRManager] ✗ Помилка виклику ocr(): {e}", flush=True)
            return self._try_predict_fallback_on_error(ocr_instance, image)
    
    def _try_predict_fallback_on_error(self, ocr_instance, image: np.ndarray):
        """Спроба використати predict() як fallback при помилці ocr()"""
        if not hasattr(ocr_instance, 'predict'):
            return None
        
        try:
            logger.info("[PaddleOCR-OCRManager] Спроба використати predict() як fallback...")
            print("[PaddleOCR-OCRManager] Спроба використати predict() як fallback...", flush=True)
            results = ocr_instance.predict(image)
            logger.info(f"[PaddleOCR-OCRManager] predict() завершено, тип результату: {type(results)}")
            print(f"[PaddleOCR-OCRManager] predict() завершено, тип результату: {type(results)}", flush=True)
            return results
        except Exception as e2:
            logger.error(f"[PaddleOCR-OCRManager] Помилка predict(): {e2}", exc_info=True)
            print(f"[PaddleOCR-OCRManager] ✗ Помилка predict(): {e2}", flush=True)
            return None
    
    def _parse_ocr_results(self, results, language: OCRLanguage) -> list[str]:
        """Парсинг результатів OCR"""
        if not isinstance(results, (list, tuple)) or len(results) == 0:
            return self._handle_empty_or_invalid_results(results)
        
        first_item = results[0]
        logger.info(f"[PaddleOCR-OCRManager] results[0] тип: {type(first_item)}")
        print(f"[PaddleOCR-OCRManager] results[0] тип: {type(first_item)}", flush=True)
        
        rec_texts, rec_scores = self._extract_rec_texts_and_scores(first_item)
        
        if rec_texts is not None:
            return self._process_rec_texts(rec_texts, rec_scores, language)
        
        if isinstance(first_item, (list, tuple)):
            return self._process_legacy_format(first_item)
        
        return self._handle_unknown_result_type(results)
    
    def _handle_empty_or_invalid_results(self, results) -> list[str]:
        """Обробка порожніх або невалідних результатів"""
        if results is None:
            logger.warning("[PaddleOCR-OCRManager] Результат None")
            print("[PaddleOCR-OCRManager] ⚠️ Результат None", flush=True)
            return []
        
        if isinstance(results, (list, tuple)) and len(results) == 0:
            logger.warning("[PaddleOCR-OCRManager] Результат: порожній список")
            print("[PaddleOCR-OCRManager] ⚠️ Результат: порожній список", flush=True)
            return []
        
        return []
    
    def _extract_rec_texts_and_scores(self, first_item):
        """Витягнення rec_texts та rec_scores з результату"""
        if hasattr(first_item, 'rec_texts'):
            rec_texts = first_item.rec_texts
            rec_scores = getattr(first_item, 'rec_scores', None)
            logger.info(f"[PaddleOCR-OCRManager] Знайдено rec_texts через атрибут: {rec_texts}")
            print(f"[PaddleOCR-OCRManager] Знайдено rec_texts через атрибут: {rec_texts}", flush=True)
            return rec_texts, rec_scores
        
        if isinstance(first_item, dict) and 'rec_texts' in first_item:
            rec_texts = first_item['rec_texts']
            rec_scores = first_item.get('rec_scores', None)
            logger.info(f"[PaddleOCR-OCRManager] Знайдено rec_texts через словник: {rec_texts}")
            print(f"[PaddleOCR-OCRManager] Знайдено rec_texts через словник: {rec_texts}", flush=True)
            logger.info(f"[PaddleOCR-OCRManager] rec_scores тип: {type(rec_scores)}, значення: {rec_scores}")
            print(f"[PaddleOCR-OCRManager] rec_scores тип: {type(rec_scores)}, значення: {rec_scores}", flush=True)
            return rec_texts, rec_scores
        
        return None, None
    
    def _process_rec_texts(self, rec_texts, rec_scores, language: OCRLanguage) -> list[str]:
        """Обробка rec_texts з фільтрацією за впевненістю"""
        logger.info(f"[PaddleOCR-OCRManager] Обробка rec_texts: {rec_texts}")
        print(f"[PaddleOCR-OCRManager] Обробка rec_texts: {rec_texts}", flush=True)
        
        if isinstance(rec_texts, str):
            text_str = rec_texts.strip()
            if text_str:
                logger.info(f"[PaddleOCR-OCRManager] Додано текст (рядок): '{text_str}'")
                print(f"[PaddleOCR-OCRManager] Додано текст (рядок): '{text_str}'", flush=True)
                return [text_str]
            return []
        
        if not isinstance(rec_texts, (list, tuple)):
            return []
        
        min_confidence = 0.3 if language == OCRLanguage.ENGLISH else 0.4
        text_parts = []
        
        for idx, text_item in enumerate(rec_texts):
            text_str = str(text_item).strip() if text_item else ""
            if not text_str:
                logger.warning(f"[PaddleOCR-OCRManager] ✗ Пропущено порожній результат (idx={idx})")
                print(f"[PaddleOCR-OCRManager] ✗ Пропущено порожній результат (idx={idx})", flush=True)
                continue
            
            if not self._should_include_text_item(text_str, rec_scores, idx, min_confidence):
                continue
            
            text_parts.append(text_str)
        
        return text_parts
    
    def _should_include_text_item(self, text_str: str, rec_scores, idx: int, min_confidence: float) -> bool:
        """Визначення, чи слід включити текстовий елемент"""
        if not rec_scores or idx >= len(rec_scores):
            logger.warning(f"[PaddleOCR-OCRManager] ⚠️ Впевненість недоступна для '{text_str}' (idx={idx}, rec_scores={rec_scores is not None})")
            print(f"[PaddleOCR-OCRManager] ⚠️ Впевненість недоступна для '{text_str}'", flush=True)
            return True
        
        score = rec_scores[idx]
        if not isinstance(score, (int, float)):
            logger.warning(f"[PaddleOCR-OCRManager] ⚠️ Впевненість не число ({type(score)}) для '{text_str}', але додаємо")
            print(f"[PaddleOCR-OCRManager] ⚠️ Впевненість не число для '{text_str}', але додаємо", flush=True)
            return True
        
        if score <= 0.0:
            logger.warning(f"[PaddleOCR-OCRManager] ✗ Пропущено '{text_str}' (нульова впевненість {score:.3f})")
            print(f"[PaddleOCR-OCRManager] ✗ Пропущено '{text_str}' (нульова впевненість {score:.3f})", flush=True)
            return False
        
        logger.info(f"[PaddleOCR-OCRManager] Текст '{text_str}' з впевненістю {score:.3f} (поріг: {min_confidence})")
        print(f"[PaddleOCR-OCRManager] Текст '{text_str}' з впевненістю {score:.3f} (поріг: {min_confidence})", flush=True)
        
        if score < min_confidence:
            logger.warning(f"[PaddleOCR-OCRManager] ✗ Пропущено '{text_str}' (низька впевненість {score:.3f} < {min_confidence})")
            print(f"[PaddleOCR-OCRManager] ✗ Пропущено '{text_str}' (низька впевненість {score:.3f} < {min_confidence})", flush=True)
            return False
        
        logger.info(f"[PaddleOCR-OCRManager] ✓ Прийнято '{text_str}' (впевненість {score:.3f} >= {min_confidence})")
        print(f"[PaddleOCR-OCRManager] ✓ Прийнято '{text_str}' (впевненість {score:.3f} >= {min_confidence})", flush=True)
        return True
    
    def _process_legacy_format(self, first_item: list) -> list[str]:
        """Обробка старого формату результатів"""
        logger.info("[PaddleOCR-OCRManager] Старий формат: список списків")
        print("[PaddleOCR-OCRManager] Старий формат: список списків", flush=True)
        text_parts = []
        
        try:
            for idx, line in enumerate(first_item):
                logger.debug("[PaddleOCR-OCRManager] Обробка рядка %s: тип=%s", idx, type(line))
                if not isinstance(line, (list, tuple)) or len(line) < 2:
                    continue
                
                text_item = line[1]
                text_str = self._extract_text_from_legacy_item(text_item)
                if text_str:
                    text_parts.append(text_str)
        except Exception as e:
            logger.error("[PaddleOCR-OCRManager] Помилка обробки старого формату: %s", e, exc_info=True)
            print(f"[PaddleOCR-OCRManager] ✗ Помилка обробки старого формату: {e}", flush=True)
        
        return text_parts
    
    def _extract_text_from_legacy_item(self, text_item) -> str:
        """Витягнення тексту з елемента старого формату"""
        if isinstance(text_item, (list, tuple)) and len(text_item) >= 1:
            text_str = str(text_item[0]).strip()
            if text_str:
                logger.debug("[PaddleOCR-OCRManager] Додано текст: '%s'", text_str)
                return text_str
        elif isinstance(text_item, str):
            text_str = text_item.strip()
            if text_str:
                logger.debug("[PaddleOCR-OCRManager] Додано текст (рядок): '%s'", text_str)
                return text_str
        return ""
    
    def _handle_unknown_result_type(self, results) -> list[str]:
        """Обробка невідомого типу результату"""
        logger.warning(f"[PaddleOCR-OCRManager] Невідомий тип результату: {type(results)}")
        print(f"[PaddleOCR-OCRManager] ⚠️ Невідомий тип результату: {type(results)}", flush=True)
        
        try:
            text = str(results).strip()
            if text and len(text) > 0:
                logger.info(f"[PaddleOCR-OCRManager] Конвертовано в рядок: {text[:100]}...")
                return [text]
        except Exception:
            pass
        
        return []
    
    def _try_predict_fallback(self, ocr_instance, image: np.ndarray) -> str:
        """Спроба використати predict() як fallback при порожньому результаті"""
        if not hasattr(ocr_instance, 'predict'):
            return ""
        
        try:
            logger.info("[PaddleOCR-OCRManager] ocr() повернув порожній результат, спроба predict() як fallback...")
            print("[PaddleOCR-OCRManager] ocr() повернув порожній результат, спроба predict() як fallback...", flush=True)
            
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
            
            return self._extract_text_from_fallback_results(fallback_results)
        except Exception as e:
            logger.warning(f"[PaddleOCR-OCRManager] predict() fallback помилка: {e}")
            print(f"[PaddleOCR-OCRManager] predict() fallback помилка: {e}", flush=True)
            return ""
    
    def _extract_text_from_fallback_results(self, fallback_results) -> str:
        """Витягнення тексту з результатів fallback"""
        if not isinstance(fallback_results, (list, tuple)) or len(fallback_results) == 0:
            return ""
        
        fallback_item = fallback_results[0]
        fallback_texts = None
        
        if hasattr(fallback_item, 'rec_texts'):
            fallback_texts = fallback_item.rec_texts
        elif isinstance(fallback_item, dict) and 'rec_texts' in fallback_item:
            fallback_texts = fallback_item['rec_texts']
        
        if fallback_texts and isinstance(fallback_texts, (list, tuple)) and len(fallback_texts) > 0:
            result_text = ' '.join(str(t).strip() for t in fallback_texts if t and str(t).strip()).strip()
            logger.info(f"[PaddleOCR-OCRManager] predict() fallback дав результат: '{result_text}'")
            print(f"[PaddleOCR-OCRManager] predict() fallback дав результат: '{result_text}'", flush=True)
            return result_text
        
        return ""
    
    def _apply_ukrainian_postprocessing(self, result_text: str) -> str:
        """Пост-обробка для української мови"""
        logger.info("[PaddleOCR-OCRManager] Застосовано пост-обробку для української мови")
        print("[PaddleOCR-OCRManager] Застосовано пост-обробку для української мови", flush=True)
        
        has_cyrillic = any('\u0400' <= char <= '\u04FF' for char in result_text)
        has_latin = any(char.isalpha() and ord(char) < 128 and not ('\u0400' <= char <= '\u04FF') for char in result_text)
        
        # Якщо вже є кирилиця, застосовуємо тільки спеціальні виправлення
        if has_cyrillic:
            corrected_text = self._apply_special_corrections(result_text)
            if corrected_text != result_text:
                logger.info(f"[PaddleOCR-OCRManager] Виправлено: '{result_text}' -> '{corrected_text}'")
                print(f"[PaddleOCR-OCRManager] Виправлено: '{result_text}' -> '{corrected_text}'", flush=True)
            return corrected_text
        
        # Якщо тільки латиниця, застосовуємо спеціальні виправлення
        corrected_text = self._apply_special_corrections(result_text)
        
        # Автоматичну конвертацію латиниці в кирилицю застосовуємо тільки для очевидних помилок OCR
        # (наприклад, коли це виглядає як помилка розпізнавання, а не як англійський текст)
        # Не застосовуємо для текстів, які виглядають як повністю неправильне розпізнавання
        if has_latin and not has_cyrillic:
            # Перевіряємо, чи текст виглядає як помилка OCR (містить змішані великі/малі літери без сенсу)
            # або як очевидні помилки (наприклад, "Tect" -> "Тест")
            if self._looks_like_ocr_error(corrected_text):
                logger.info(f"[PaddleOCR-OCRManager] has_cyrillic={has_cyrillic}, has_latin={has_latin}, застосовуємо пост-обробку")
                print(f"[PaddleOCR-OCRManager] has_cyrillic={has_cyrillic}, has_latin={has_latin}, застосовуємо пост-обробку", flush=True)
                corrected_text = self._apply_latin_to_cyrillic_replacements(corrected_text)
            else:
                logger.info(f"[PaddleOCR-OCRManager] Текст '{corrected_text}' не виглядає як помилка OCR, пропускаємо конвертацію")
                print(f"[PaddleOCR-OCRManager] Текст '{corrected_text}' не виглядає як помилка OCR, пропускаємо конвертацію", flush=True)
        
        if corrected_text != result_text:
            logger.info(f"[PaddleOCR-OCRManager] Виправлено: '{result_text}' -> '{corrected_text}'")
            print(f"[PaddleOCR-OCRManager] Виправлено: '{result_text}' -> '{corrected_text}'", flush=True)
            return corrected_text
        
        return result_text
    
    def _looks_like_ocr_error(self, text: str) -> bool:
        """
        Перевірка, чи текст виглядає як помилка OCR (а не як англійський текст)
        
        Args:
            text: текст для перевірки
            
        Returns:
            True, якщо текст виглядає як помилка OCR
        """
        if not text or len(text.strip()) < 2:
            return False
        
        text_lower = text.lower().strip()
        
        # Якщо текст містить змішані великі/малі літери без сенсу (наприклад, "vaT HAJ")
        # або виглядає як очевидні помилки OCR
        has_mixed_case = any(c.isupper() for c in text) and any(c.islower() for c in text)
        
        # Перевіряємо, чи це виглядає як англійське слово (якщо так, не конвертуємо)
        # Прості англійські слова, які не потрібно конвертувати
        common_english_words = ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use']
        
        # Якщо текст схожий на англійське слово, не конвертуємо
        words = text_lower.split()
        if len(words) == 1 and text_lower in common_english_words:
            return False
        
        # Якщо текст містить змішані великі/малі літери без сенсу, це може бути помилка OCR
        if has_mixed_case and len(text.strip()) <= 10:
            # Перевіряємо, чи це не виглядає як англійське слово з великої літери
            if text[0].isupper() and text[1:].islower() and len(text) > 3:
                # Може бути англійське слово, не конвертуємо
                return False
            return True
        
        # Якщо текст дуже короткий (1-3 символи) і містить тільки латиницю, це може бути помилка
        if len(text.strip()) <= 3 and text.isalpha():
            return True
        
        # Для довших текстів не конвертуємо автоматично
        return False
    
    def _apply_special_corrections(self, text: str) -> str:
        """Застосування спеціальних виправлень"""
        corrected = text
        
        if 'pubi' in corrected.lower():
            corrected = corrected.replace('pubi', 'рив')
            corrected = corrected.replace('Pubi', 'Рив')
            corrected = corrected.replace('PUBI', 'РИВ')
        
        if corrected.lower() == 'tect':
            corrected = 'Тест'
            logger.info("[PaddleOCR-OCRManager] Спеціальна заміна: 'Tect' -> 'Тест'")
            print("[PaddleOCR-OCRManager] Спеціальна заміна: 'Tect' -> 'Тест'", flush=True)
        
        if '&' in corrected:
            corrected = corrected.replace('&', '')
            logger.info("[PaddleOCR-OCRManager] Видалено спеціальний символ '&'")
            print("[PaddleOCR-OCRManager] Видалено спеціальний символ '&'", flush=True)
        
        if '6' in corrected:
            corrected = corrected.replace('6', 'б')
            logger.info("[PaddleOCR-OCRManager] Виправлено '6' -> 'б'")
            print("[PaddleOCR-OCRManager] Виправлено '6' -> 'б'", flush=True)
        
        if '0' in corrected:
            corrected = corrected.replace('0', 'о')
            logger.info("[PaddleOCR-OCRManager] Виправлено '0' -> 'о'")
            print("[PaddleOCR-OCRManager] Виправлено '0' -> 'о'", flush=True)
        
        return corrected
    
    def _apply_latin_to_cyrillic_replacements(self, text: str) -> str:
        """Застосування замін латинських літер на кириличні"""
        replacements = [
            ('T', 'Т'), ('t', 'т'), ('E', 'Е'), ('e', 'е'), ('C', 'С'), ('c', 'с'),
            ('r', 'р'), ('I', 'І'), ('i', 'і'), ('n', 'н'), ('d', 'д'), ('B', 'В'),
            ('p', 'р'), ('u', 'в'), ('O', 'О'), ('o', 'о'), ('A', 'А'), ('a', 'а'),
            ('H', 'Н'), ('Y', 'У'), ('X', 'Х'), ('M', 'М'), ('P', 'Р'), ('K', 'К'),
            ('k', 'к'), ('v', 'в'), ('V', 'В'), ('S', 'С'), ('s', 'с'), ('L', 'Л'),
            ('l', 'л'), ('g', 'г'), ('G', 'Г'), ('f', 'в'), ('F', 'В'), ('N', 'Л'),
        ]
        
        for latin, cyrillic in replacements:
            if latin in text:
                text = text.replace(latin, cyrillic)
        
        return text
    
    def _log_final_result(self, result_text: str, results) -> None:
        """Логування фінального результату"""
        logger.info(f"[PaddleOCR-OCRManager] Фінальний результат: {len(result_text)} символів")
        print(f"[PaddleOCR-OCRManager] Фінальний результат: {len(result_text)} символів", flush=True)
        
        if result_text:
            logger.info(f"[PaddleOCR-OCRManager] Перші 100 символів: {result_text[:100]}...")
            print(f"[PaddleOCR-OCRManager] Перші 100 символів: {result_text[:100]}...", flush=True)
        else:
            logger.warning("[PaddleOCR-OCRManager] ⚠️ ФІНАЛЬНИЙ РЕЗУЛЬТАТ ПОРОЖНІЙ!")
            print("[PaddleOCR-OCRManager] ⚠️ ФІНАЛЬНИЙ РЕЗУЛЬТАТ ПОРОЖНІЙ!", flush=True)
            print(f"[PaddleOCR-OCRManager] Повний результат для діагностики: {results}", flush=True)


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
