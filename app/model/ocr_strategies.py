"""
OCR Strategies - Реалізація патерну Strategy для різних OCR рушіїв
handwrite2print/app/model/ocr_strategies.py
"""
import sys
import os
from pathlib import Path

# КРИТИЧНО: Додавання локальної версії PaddleOCR до sys.path ДО будь-яких імпортів
# Це дозволяє використовувати локальну версію замість pip-встановленої
try:
    current_file = Path(__file__).resolve()
    # app/model/ocr_strategies.py -> app -> resources/models/v3.3.2 source code/PaddlePaddle-PaddleOCR-95dc316
    base_dir = current_file.parent.parent.parent
    local_paddleocr_dir = base_dir / "resources" / "models" / "v3.3.2 source code" / "PaddlePaddle-PaddleOCR-95dc316"
    
    if local_paddleocr_dir.exists() and (local_paddleocr_dir / "paddleocr" / "__init__.py").exists():
        local_paddleocr_path = str(local_paddleocr_dir.absolute())
        if local_paddleocr_path not in sys.path:
            # ВАЖЛИВО: Додаємо на початок, щоб локальна версія мала пріоритет
            sys.path.insert(0, local_paddleocr_path)
            # Логуємо тільки після налаштування логування
            import logging
            temp_logger = logging.getLogger(__name__)
            temp_logger.info(f"[PaddleOCR] Локальна версія PaddleOCR додана до sys.path: {local_paddleocr_path}")
except Exception as e:
    # Логуємо тільки після налаштування логування
    import logging
    temp_logger = logging.getLogger(__name__)
    temp_logger.debug(f"[PaddleOCR] Локальна версія не знайдена: {e}")

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Глобальне кешування статусу доступності рушіїв
_ENGINE_AVAILABILITY_CACHE = {}

# Константи для мов
LANG_ENG_UKR = 'eng+ukr'
LANG_UKR_ENG = 'ukr+eng'


class OCRStrategy(ABC):
    """Абстрактний базовий клас для OCR стратегій"""
    
    @abstractmethod
    def recognize(self, image: np.ndarray, language: str) -> str:
        """
        Розпізнавання тексту з зображення
        
        Args:
            image: оброблене зображення (numpy array)
            language: код мови (eng, ukr, eng+ukr)
            
        Returns:
            розпізнаний текст
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Перевірка доступності рушія"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Повертає назву рушія"""
        pass


class TesseractStrategy(OCRStrategy):
    """Стратегія для Tesseract OCR"""
    
    def __init__(self):
        self._initialized = False
        self._available = False
        self._init()
    
    def _init(self):
        """Ініціалізація Tesseract"""
        try:
            import pytesseract
            import os
            
            # Автоматичне налаштування шляху
            try:
                pytesseract.get_tesseract_version()
                self._available = True
            except Exception:
                paths = [
                    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                    os.path.expandvars(r'C:\Users\%USERNAME%\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'),
                ]
                for path in paths:
                    if os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        self._available = True
                        break
            
            self._initialized = True
        except Exception as e:
            logger.warning(f"Tesseract не доступний: {e}")
            self._available = False
    
    def is_available(self) -> bool:
        return self._available
    
    def get_name(self) -> str:
        return "Tesseract"
    
    def recognize(self, image: np.ndarray, language: str) -> str:
        """Розпізнавання через Tesseract"""
        if not self._available:
            raise RuntimeError("Tesseract не доступний")
        
        # Валідація вхідного зображення
        if image is None:
            raise ValueError("Зображення не може бути None")
        if image.size == 0:
            raise ValueError("Зображення порожнє")
        if len(image.shape) < 2 or len(image.shape) > 3:
            raise ValueError(f"Невірна розмірність зображення: {image.shape}")
        
        try:
            import pytesseract
            import cv2
            
            # Переконуємося, що зображення в правильному форматі
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            # Покращення контрасту для Tesseract
            if len(image.shape) == 2:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                image = clahe.apply(image)
            
            # Мапінг мов
            lang_map = {
                'eng': 'eng', 
                'ukr': 'ukr', 
                LANG_ENG_UKR: LANG_ENG_UKR,
                LANG_UKR_ENG: LANG_UKR_ENG,
            }
            tesseract_lang = lang_map.get(language, 'eng')
            
            logger.info(f"[Tesseract] Використовується мова: {tesseract_lang} (запитана: {language})")
            
            # Пробуємо різні PSM режими для кращого результату
            # Для рукописного тексту найкращі: 6 (блок тексту), 7 (один рядок), 8 (слово), 11 (розрізнений текст)
            text = ""
            psm_modes = [6, 7, 8, 11, 13]  # Найкращі для рукопису
            
            for psm_mode in psm_modes:
                try:
                    custom_config = f'--oem 3 --psm {psm_mode}'
                    test_text = pytesseract.image_to_string(image, lang=tesseract_lang, config=custom_config)
                    logger.debug(f"[Tesseract] PSM {psm_mode}: '{test_text.strip()}'")
                    if len(test_text.strip()) > len(text.strip()):
                        text = test_text
                except Exception as e:
                    logger.debug(f"[Tesseract] PSM {psm_mode} помилка: {e}")
                    continue
            
            # Fallback на базовий режим
            if not text or len(text.strip()) < 1:
                try:
                    custom_config = r'--oem 3 --psm 6'
                    text = pytesseract.image_to_string(image, lang=tesseract_lang, config=custom_config)
                    logger.debug(f"[Tesseract] Fallback PSM 6: '{text.strip()}'")
                except Exception as e:
                    logger.warning(f"[Tesseract] Fallback помилка: {e}")
                    text = ""
            
            
            result = text.strip()
            logger.info(f"[Tesseract] Фінальний результат ({len(result)} символів): '{result}'")
            return result
            
        except Exception as e:
            logger.error(f"Помилка Tesseract: {e}")
            raise


class EasyOCRStrategy(OCRStrategy):
    """Стратегія для EasyOCR"""
    
    def __init__(self):
        self._reader = None
        self._available = False
        self._init()
    
    def _init(self):
        """Лінива ініціалізація EasyOCR"""
        global _ENGINE_AVAILABILITY_CACHE
        
        # Перевірка кешу
        cache_key = 'easyocr'
        if cache_key in _ENGINE_AVAILABILITY_CACHE:
            self._available = _ENGINE_AVAILABILITY_CACHE[cache_key]
            return
        
        # Використовуємо покращену перевірку PyTorch
        try:
            from .pytorch_helper import check_pytorch_availability, setup_pytorch_path
            
            # Налаштовуємо PATH (ДУЖЕ ВАЖЛИВО - має бути перед будь-яким імпортом torch)
            setup_pytorch_path()
            
            # Невелика затримка для завантаження DLL
            import time
            time.sleep(0.1)
            
            # Перевірка PyTorch
            is_available, version, _ = check_pytorch_availability()
            
            if is_available:
                self._available = True
                _ENGINE_AVAILABILITY_CACHE[cache_key] = True
                logger.debug(f"EasyOCR доступний (PyTorch {version})")
            else:
                # Логуємо тільки один раз, зменшуємо рівень до INFO
                if cache_key not in _ENGINE_AVAILABILITY_CACHE:
                    # Спрощене повідомлення - детальна інформація в FINAL_SOLUTION_PYTORCH.md
                    logger.info("EasyOCR не доступний через проблеми з PyTorch DLL. Використовується Tesseract.")
                self._available = False
                _ENGINE_AVAILABILITY_CACHE[cache_key] = False
        except ImportError:
            # Якщо pytorch_helper недоступний, використовуємо базову перевірку
            try:
                import torch
                self._available = True
                _ENGINE_AVAILABILITY_CACHE[cache_key] = True
            except Exception as e:
                if cache_key not in _ENGINE_AVAILABILITY_CACHE:
                    logger.warning(f"EasyOCR не доступний (PyTorch): {e}")
                self._available = False
                _ENGINE_AVAILABILITY_CACHE[cache_key] = False
        except Exception as e:
            if cache_key not in _ENGINE_AVAILABILITY_CACHE:
                logger.warning(f"EasyOCR не доступний: {e}")
            self._available = False
            _ENGINE_AVAILABILITY_CACHE[cache_key] = False
    
    def _get_reader(self, language: str = 'eng'):
        """Отримання Reader (лінива ініціалізація з кешуванням по мовах)"""
        # Кешуємо Reader по мовах для кращої продуктивності
        if not hasattr(self, '_readers'):
            self._readers = {}
        
        # Мапінг мов для EasyOCR
        # Для української використовуємо тільки 'uk' для кращого розпізнавання кирилиці
        lang_map = {
            'eng': ['en'],
            'ukr': ['uk'],  # Тільки українська для кращого розпізнавання
            LANG_ENG_UKR: ['en', 'uk'],
            LANG_UKR_ENG: ['uk', 'en']
        }
        easyocr_langs = lang_map.get(language, ['en'])
        lang_key = ','.join(easyocr_langs)  # Використовуємо як ключ кешу
        
        if lang_key not in self._readers:
            try:
                # Спочатку перевіряємо PyTorch
                from .pytorch_helper import setup_pytorch_path, check_pytorch_availability
                setup_pytorch_path()
                is_available, _, _ = check_pytorch_availability()
                
                if not is_available:
                    raise RuntimeError("PyTorch не доступний")
                
                import easyocr
                logger.info(f"[EasyOCR] Створення Reader для мов: {easyocr_langs}")
                self._readers[lang_key] = easyocr.Reader(easyocr_langs, gpu=False)  # CPU only
                logger.info(f"[EasyOCR] Reader створено для мов: {easyocr_langs}")
            except ImportError as e:
                logger.error(f"EasyOCR не встановлено: {e}")
                raise RuntimeError("EasyOCR не встановлено. Встановіть: pip install easyocr")
            except Exception as e:
                error_msg = str(e)
                if "DLL" in error_msg or "1114" in error_msg:
                    from .pytorch_helper import get_pytorch_error_solution
                    logger.error(f"Помилка PyTorch DLL при ініціалізації EasyOCR: {e}")
                    logger.error(get_pytorch_error_solution())
                else:
                    logger.error(f"Помилка ініціалізації EasyOCR: {e}")
                raise
        
        return self._readers[lang_key]
    
    def is_available(self) -> bool:
        return self._available
    
    def get_name(self) -> str:
        return "EasyOCR"
    
    def recognize(self, image: np.ndarray, language: str) -> str:
        """Розпізнавання через EasyOCR"""
        if not self._available:
            raise RuntimeError("EasyOCR не доступний")
        
        try:
            import cv2
            reader = self._get_reader(language)
            processed_image = self._prepare_image_for_easyocr(image, language)
            easyocr_langs = self._get_easyocr_languages(language)
            
            result = self._perform_easyocr_recognition(reader, processed_image, easyocr_langs, language)
            filtered_result = self._filter_and_sort_results(result, language)
            return self._extract_text_from_results(filtered_result)
            
        except Exception as e:
            logger.error(f"Помилка EasyOCR: {e}")
            raise
    
    def _prepare_image_for_easyocr(self, image: np.ndarray, language: str) -> np.ndarray:
        """Підготовка зображення для EasyOCR"""
        import cv2
        
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        logger.info(f"[EasyOCR] Використання мінімальної обробки для мови: {language}")
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            logger.info("[EasyOCR] Конвертовано grayscale -> RGB")
        elif len(image.shape) == 3:
            image = self._convert_3channel_to_rgb(image)
        
        logger.info("[EasyOCR] Використовується оригінальне зображення з мінімальною обробкою")
        return image
    
    def _convert_3channel_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """Конвертація 3-канального зображення в RGB"""
        import cv2
        
        if image.shape[2] == 1:
            image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
            logger.info("[EasyOCR] Конвертовано 1 канал -> RGB")
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.info("[EasyOCR] Конвертовано BGR -> RGB")
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            logger.info("[EasyOCR] Конвертовано RGBA -> RGB")
        
        return image
    
    def _get_easyocr_languages(self, language: str) -> list[str]:
        """Отримання списку мов для EasyOCR"""
        lang_map = {
            'eng': ['en'],
            'ukr': ['uk'],
            LANG_ENG_UKR: ['en', 'uk'],
            LANG_UKR_ENG: ['uk', 'en']
        }
        easyocr_langs = lang_map.get(language, ['en'])
        logger.info(f"[EasyOCR] Використовується мова: {easyocr_langs}")
        return easyocr_langs
    
    def _perform_easyocr_recognition(self, reader, image: np.ndarray, easyocr_langs: list[str], language: str):
        """Виконання розпізнавання через EasyOCR"""
        width_ths = 0.5
        height_ths = 0.5
        paragraph = False
        
        logger.info(f"[EasyOCR] Параметри: width_ths={width_ths}, height_ths={height_ths}, мова={language}")
        
        result = reader.readtext(
            image,
            paragraph=paragraph,
            detail=1,
            width_ths=width_ths,
            height_ths=height_ths,
            allowlist=None,
            blocklist=None
        )
        
        logger.info(f"[EasyOCR] Отримано {len(result)} результатів")
        if result:
            logger.info(f"[EasyOCR] Перші 3 результати: {result[:3]}")
        
        return result
    
    def _filter_and_sort_results(self, result: list, language: str) -> list:
        """Фільтрація та сортування результатів EasyOCR"""
        confidence_threshold = 0.05
        filtered_result = []
        
        for idx, detection in enumerate(result):
            if not self._is_valid_detection(detection):
                continue
            
            bbox, txt, conf = detection[0], detection[1], detection[2]
            logger.info(f"[EasyOCR] Результат {idx}: txt='{txt}' (тип: {type(txt)}), conf={conf} (тип: {type(conf)})")
            
            if not self._should_include_detection(txt, conf, confidence_threshold, idx):
                continue
            
            filtered_result.append((bbox, txt, conf))
        
        logger.info(f"[EasyOCR] Відфільтровано {len(filtered_result)} результатів з {len(result)}")
        
        if filtered_result:
            self._sort_results_by_position(filtered_result)
            self._log_confidence_stats(filtered_result)
        
        return filtered_result
    
    def _is_valid_detection(self, detection) -> bool:
        """Перевірка валідності детекції"""
        if not isinstance(detection, (list, tuple)) or len(detection) < 3:
            logger.warning(f"[EasyOCR] Невірний формат результату: {detection} (тип: {type(detection)}, довжина: {len(detection) if hasattr(detection, '__len__') else 'N/A'})")
            return False
        return True
    
    def _should_include_detection(self, txt, conf, confidence_threshold: float, idx: int) -> bool:
        """Визначення, чи слід включити детекцію в результат"""
        if not txt or (isinstance(txt, str) and not txt.strip()):
            logger.warning(f"[EasyOCR] ✗ Пропущено (порожній текст): '{txt}'")
            return False
        
        if not isinstance(conf, (int, float)):
            logger.warning(f"[EasyOCR] ⚠️ Впевненість не число ({type(conf)}), але додаємо результат: '{txt}'")
            return True
        
        if conf > confidence_threshold:
            if conf < 0.3:
                logger.warning(f"[EasyOCR] ⚠️ Додано результат з низькою впевненістю: '{txt}' (conf={conf:.3f}) - можлива помилка розпізнавання")
            else:
                logger.info(f"[EasyOCR] ✓ Додано: '{txt}' (conf={conf:.3f})")
            return True
        
        logger.info(f"[EasyOCR] ✗ Пропущено (низька впевненість {conf:.3f} <= {confidence_threshold}): '{txt}'")
        return False
    
    def _sort_results_by_position(self, filtered_result: list) -> None:
        """Сортування результатів за позицією"""
        filtered_result.sort(key=lambda x: (
            x[0][0][1] if isinstance(x[0], (list, tuple)) and len(x[0]) > 0 else 0,
            x[0][0][0] if isinstance(x[0], (list, tuple)) and len(x[0]) > 0 else 0
        ))
    
    def _log_confidence_stats(self, filtered_result: list) -> None:
        """Логування статистики впевненості"""
        confidences = [conf for _, _, conf in filtered_result if isinstance(conf, (int, float))]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            min_confidence = min(confidences)
            logger.info(f"[EasyOCR] Якість розпізнавання: середня впевненість={avg_confidence:.3f}, мінімальна={min_confidence:.3f}")
            if min_confidence < 0.3:
                logger.warning("[EasyOCR] ⚠️ Увага: є результати з дуже низькою впевненістю (< 0.3) - можливі помилки розпізнавання")
    
    def _extract_text_from_results(self, filtered_result: list) -> str:
        """Витягнення тексту з відфільтрованих результатів"""
        text = ' '.join([txt for _, txt, _ in filtered_result])
        return text.strip()


class PaddleOCRStrategy(OCRStrategy):
    """Стратегія для PaddleOCR"""
    
    def __init__(self):
        self._instances = {}  # Кеш екземплярів по мовах
        self._available = False
        self._creating_instances = {}  # Захист від одночасного створення
        self._init()
    
    def _check_pytorch_availability(self) -> tuple[bool, str]:
        """Перевірка доступності PyTorch"""
        from .pytorch_helper import check_pytorch_availability, setup_pytorch_path
        import time

        setup_pytorch_path()
        time.sleep(0.1)

        pytorch_available, pytorch_version, _ = check_pytorch_availability()
        return pytorch_available, pytorch_version or ""
    
    def _check_paddlepaddle_availability(self, pytorch_version: str) -> bool:
        """Перевірка доступності PaddlePaddle"""
        try:
            logger.info("[PaddleOCR] Спробуємо імпортувати paddle...")
            import time
            import_start = time.time()
            import paddle  # type: ignore[import]
            import_elapsed = time.time() - import_start
            logger.info(
                "[PaddleOCR] paddle успішно імпортовано за %.2f секунд",
                import_elapsed,
            )
            logger.info(
                "[PaddleOCR] paddle доступний, paddleocr буде перевірено при створенні екземпляра",
            )
            logger.info(
                "[PaddleOCR] PaddleOCR доступний (PyTorch %s)", pytorch_version
            )
            return True
        except ImportError as e:
            logger.warning(
                "PaddleOCR не доступний: PaddlePaddle не встановлено: %s", e
            )
            return False
        except Exception as e:
            logger.warning("PaddleOCR не доступний: %s", e)
            return False
    
    def _init(self):
        """Лінива ініціалізація PaddleOCR - тільки перевірка доступності, без створення моделей"""
        global _ENGINE_AVAILABILITY_CACHE
        
        cache_key = 'paddleocr'
        if cache_key in _ENGINE_AVAILABILITY_CACHE:
            self._available = _ENGINE_AVAILABILITY_CACHE[cache_key]
            return
        
        try:
            pytorch_available, pytorch_version = self._check_pytorch_availability()
            if not pytorch_available:
                if cache_key not in _ENGINE_AVAILABILITY_CACHE:
                    logger.info("PaddleOCR не доступний через проблеми з PyTorch DLL. Використовується Tesseract.")
                self._available = False
                _ENGINE_AVAILABILITY_CACHE[cache_key] = False
                return
                
            paddle_available = self._check_paddlepaddle_availability(pytorch_version)
            self._available = paddle_available
            _ENGINE_AVAILABILITY_CACHE[cache_key] = paddle_available
        except ImportError:
            # Якщо pytorch_helper недоступний, використовуємо базову перевірку
            try:
                import torch
                self._available = True
                _ENGINE_AVAILABILITY_CACHE[cache_key] = True
            except Exception as e:
                if cache_key not in _ENGINE_AVAILABILITY_CACHE:
                    logger.warning(f"PaddleOCR не доступний: {e}")
                self._available = False
                _ENGINE_AVAILABILITY_CACHE[cache_key] = False
        except Exception as e:
            if cache_key not in _ENGINE_AVAILABILITY_CACHE:
                logger.warning(f"PaddleOCR не доступний: {e}")
            self._available = False
            _ENGINE_AVAILABILITY_CACHE[cache_key] = False
    
    def _get_instance(self, language: str = 'en'):
        """Отримання екземпляра PaddleOCR (лінива ініціалізація з кешуванням по мовах)"""
        paddle_lang = self._map_language_to_paddle(language)
        
        cached_instance = self._get_cached_instance(paddle_lang)
        if cached_instance:
            return cached_instance
        
        if paddle_lang in self._creating_instances:
            return self._wait_for_instance_creation(paddle_lang)
        
        return self._create_new_instance(paddle_lang)
    
    def _map_language_to_paddle(self, language: str) -> str:
        """Мапінг мови на код PaddleOCR"""
        lang_map = {
            'eng': 'en',
            'ukr': 'uk',
            LANG_ENG_UKR: 'en',
            LANG_UKR_ENG: 'en'
        }
        return lang_map.get(language, 'en')
    
    def _get_cached_instance(self, paddle_lang: str):
        """Отримання закешованого екземпляра"""
        if paddle_lang in self._instances:
            logger.info(f"[PaddleOCR] ✓ Використовується закешований екземпляр для мови: {paddle_lang}")
            return self._instances[paddle_lang]
        return None
    
    def _wait_for_instance_creation(self, paddle_lang: str):
        """Очікування завершення створення екземпляра іншим потоком"""
        import time
        max_wait = 60
        wait_time = 0
        
        while paddle_lang in self._creating_instances and wait_time < max_wait:
            time.sleep(0.5)
            wait_time += 0.5
            if paddle_lang in self._instances:
                return self._instances[paddle_lang]
        
        if paddle_lang not in self._instances:
            raise RuntimeError(f"Не вдалося створити екземпляр PaddleOCR для мови {paddle_lang}")
        return self._instances[paddle_lang]
    
    def _create_new_instance(self, paddle_lang: str):
        """Створення нового екземпляра PaddleOCR"""
        self._creating_instances[paddle_lang] = True
        try:
            self._check_paddle_imports()
            instance = self._create_paddleocr_instance(paddle_lang)
            self._verify_instance(instance)
            self._cache_instance(paddle_lang, instance)
            return instance
        except Exception as e:
            self._log_instance_creation_error(paddle_lang, e)
            raise
        finally:
            if paddle_lang in self._creating_instances:
                del self._creating_instances[paddle_lang]
    
    def _check_paddle_imports(self):
        """Перевірка доступності paddle та paddleocr"""
        logger.info("[PaddleOCR] Перевірка доступності paddle та paddleocr...")
        
        try:
            import paddle
            logger.info("[PaddleOCR] paddle успішно імпортовано")
        except ImportError as e:
            logger.error(f"[PaddleOCR] paddle не встановлено: {e}")
            raise RuntimeError(f"PaddlePaddle не встановлено: {e}")
        except Exception as e:
            logger.warning(f"[PaddleOCR] Помилка імпорту paddle: {e}")
        
        self._check_paddleocr_import()
    
    def _check_paddleocr_import(self):
        """Перевірка імпорту paddleocr та визначення джерела"""
        try:
            import paddleocr
            self._log_paddleocr_source(paddleocr)
            if hasattr(paddleocr, '__version__'):
                logger.info(f"[PaddleOCR] Версія PaddleOCR: {paddleocr.__version__}")
        except Exception as e:
            logger.error(f"[PaddleOCR] Не вдалося імпортувати paddleocr: {e}")
            raise
    
    def _log_paddleocr_source(self, paddleocr):
        """Логування джерела PaddleOCR (локальна або pip)"""
        paddleocr_file = getattr(paddleocr, '__file__', None)
        if not paddleocr_file:
            return
        
        paddleocr_path = Path(paddleocr_file).resolve()
        current_file = Path(__file__).resolve()
        base_dir_check = current_file.parent.parent.parent
        local_paddleocr_dir = base_dir_check / "resources" / "models" / "v3.3.2 source code" / "PaddlePaddle-PaddleOCR-95dc316"
        
        if str(local_paddleocr_dir.absolute()) in str(paddleocr_path):
            logger.info(f"[PaddleOCR] ✓ Використовується ЛОКАЛЬНА версія PaddleOCR з: {paddleocr_path}")
        else:
            logger.info(f"[PaddleOCR] Використовується pip-встановлена версія PaddleOCR з: {paddleocr_path}")
    
    def _create_paddleocr_instance(self, paddle_lang: str):
        """Створення екземпляра PaddleOCR з обробкою помилок"""
        from paddleocr import PaddleOCR
        import time
        
        self._log_instance_creation_start(paddle_lang)
        start_time = time.time()
        
        try:
            logger.info(f"[PaddleOCR] Виклик PaddleOCR(lang='{paddle_lang}', use_angle_cls=False)...")
            logger.info("[PaddleOCR] Це може зайняти кілька хвилин при першому запуску...")
            instance = PaddleOCR(lang=paddle_lang, use_angle_cls=False)
            elapsed_time = time.time() - start_time
            logger.info(f"[PaddleOCR] ✓ Екземпляр успішно створено за {elapsed_time:.2f} секунд ({elapsed_time/60:.1f} хвилин)")
            return instance
        except Exception as init_error:
            if paddle_lang == 'uk':
                return self._try_fallback_to_ru(paddle_lang, start_time, init_error)
            self._log_instance_creation_error(paddle_lang, init_error, start_time)
            raise
    
    def _log_instance_creation_start(self, paddle_lang: str):
        """Логування початку створення екземпляра"""
        logger.info(f"[PaddleOCR] ===== ПОЧАТОК створення екземпляра для мови: {paddle_lang} =====")
        logger.info(f"[PaddleOCR] PADDLE_PDX_CACHE_HOME: {os.environ.get('PADDLE_PDX_CACHE_HOME', 'не встановлено')}")
        logger.info(f"[PaddleOCR] PADDLEX_HOME: {os.environ.get('PADDLEX_HOME', 'не встановлено')}")
    
    def _try_fallback_to_ru(self, original_lang: str, start_time, init_error):
        """Спроба fallback на 'ru' для української мови"""
        import time
        from paddleocr import PaddleOCR
        
        logger.warning(f"[PaddleOCR] Мова 'uk' не підтримується, спробуємо 'ru' для кирилиці: {init_error}")
        
        try:
            paddle_lang = 'ru'
            logger.info(f"[PaddleOCR] Виклик PaddleOCR(lang='{paddle_lang}', use_angle_cls=False)...")
            instance = PaddleOCR(lang=paddle_lang, use_angle_cls=False)
            elapsed_time = time.time() - start_time
            logger.info(f"[PaddleOCR] ✓ Екземпляр успішно створено з 'ru' за {elapsed_time:.2f} секунд")
            return instance
        except Exception as fallback_error:
            elapsed_time = time.time() - start_time
            logger.error(f"[PaddleOCR] ✗ Помилка при створенні екземпляра з 'ru' (через {elapsed_time:.2f} сек): {fallback_error}")
            import traceback
            logger.error(f"[PaddleOCR] Traceback: {traceback.format_exc()}")
            raise
    
    def _verify_instance(self, instance):
        """Перевірка створеного екземпляра"""
        try:
            logger.info("[PaddleOCR] Перевірка екземпляра...")
            logger.info(f"[PaddleOCR] Тип: {type(instance).__name__}")
            logger.info(f"[PaddleOCR] Має метод ocr: {hasattr(instance, 'ocr')}")
            logger.info(f"[PaddleOCR] Має метод predict: {hasattr(instance, 'predict')}")
        except Exception as e:
            logger.warning(f"[PaddleOCR] Не вдалося перевірити екземпляр: {e}")
    
    def _cache_instance(self, paddle_lang: str, instance):
        """Кешування екземпляра"""
        self._instances[paddle_lang] = instance
        logger.info(f"[PaddleOCR] ✓ Екземпляр закешовано для мови: {paddle_lang}")
        logger.info(f"[PaddleOCR] ===== ЗАВЕРШЕНО створення екземпляра для мови: {paddle_lang} =====")
    
    def _log_instance_creation_error(self, paddle_lang: str, error: Exception, start_time=None):
        """Логування помилки створення екземпляра"""
        import time
        import traceback
        
        if start_time:
            elapsed_time = time.time() - start_time
            logger.error(f"[PaddleOCR] ✗ Помилка при створенні екземпляра (через {elapsed_time:.2f} сек): {error}")
        else:
            logger.error(f"[PaddleOCR] Помилка ініціалізації для мови {paddle_lang}: {error}")
        logger.error(f"[PaddleOCR] Traceback: {traceback.format_exc()}")
    
    def is_available(self) -> bool:
        return self._available
    
    def get_name(self) -> str:
        return "PaddleOCR"
    
    def recognize(self, image: np.ndarray, language: str) -> str:
        """Розпізнавання через PaddleOCR"""
        self._log_recognition_start(image, language)
        
        if not self._available:
            error_msg = "PaddleOCR не доступний!"
            print(f"[PaddleOCR] ✗ {error_msg}")
            logger.error(f"[PaddleOCR] ✗ {error_msg}")
            raise RuntimeError(error_msg)
        
        try:
            import cv2
            import time
            
            recognize_start = time.time()
            self._log_recognition_begin(language, image)
            
            ocr_instance = self._get_instance(language)
            get_instance_time = time.time() - recognize_start
            logger.info(f"[PaddleOCR] Екземпляр отримано за {get_instance_time:.2f} сек")
            
            paddle_lang = self._get_paddle_language(language)
            logger.info(f"[PaddleOCR] Використовується PaddleOCR екземпляр для мови: {paddle_lang} (запитана мова: {language})")
            
            processed_image = self._prepare_image_for_paddleocr(image)
            results = self._perform_paddleocr_recognition(ocr_instance, processed_image, recognize_start)
            
            if results is None:
                results = self._try_fallback_for_none_result(ocr_instance, processed_image)
                if results is None:
                    return ""
            
            if isinstance(results, (list, tuple)) and len(results) == 0:
                results = self._try_fallback_for_empty_result(ocr_instance, processed_image)
                if not results or (isinstance(results, (list, tuple)) and len(results) == 0):
                    return ""
            
            text = self._extract_text_from_results(results)
            result_text = text.strip() if text else ""
            
            total_time = time.time() - recognize_start
            self._log_recognition_complete(result_text, results, total_time)
            
            return result_text
        except Exception as e:
            logger.error(f"[PaddleOCR] ✗ Помилка розпізнавання: {e}", exc_info=True)
            raise
    
    def _log_recognition_start(self, image: np.ndarray, language: str) -> None:
        """Логування початку розпізнавання"""
        from pathlib import Path
        debug_file = Path(__file__).parent.parent.parent / "paddleocr_debug.log"
        
        try:
            with open(debug_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write("[PaddleOCR] ===== ВИКЛИК recognize() =====\n")
                f.write(f"[PaddleOCR] Час: {__import__('datetime').datetime.now()}\n")
                f.write(f"[PaddleOCR] self._available = {self._available}\n")
                f.write(f"[PaddleOCR] Мова: {language}\n")
                f.write(f"[PaddleOCR] Розмір зображення: {image.shape if hasattr(image, 'shape') else 'N/A'}\n")
                f.flush()
        except Exception:
            pass
        
        try:
            import sys
            sys.stdout.write("[PaddleOCR] ===== ВИКЛИК recognize() =====\n")
            sys.stdout.flush()
            print("[PaddleOCR] ===== ВИКЛИК recognize() =====", flush=True)
            logger.info("[PaddleOCR] ===== ВИКЛИК recognize() =====")
            print(f"[PaddleOCR] self._available = {self._available}", flush=True)
            logger.info(f"[PaddleOCR] self._available = {self._available}")
        except Exception:
            pass
    
    def _log_recognition_begin(self, language: str, image: np.ndarray) -> None:
        """Логування початку процесу розпізнавання"""
        print("[PaddleOCR] ===== ПОЧАТОК розпізнавання =====")
        logger.info("[PaddleOCR] ===== ПОЧАТОК розпізнавання =====")
        print(f"[PaddleOCR] Мова: {language}, розмір зображення: {image.shape}")
        logger.info(f"[PaddleOCR] Мова: {language}, розмір зображення: {image.shape}")
        print(f"[PaddleOCR] Тип зображення: {type(image)}, dtype: {image.dtype if hasattr(image, 'dtype') else 'N/A'}")
        logger.info(f"[PaddleOCR] Тип зображення: {type(image)}, dtype: {image.dtype if hasattr(image, 'dtype') else 'N/A'}")
        print(f"[PaddleOCR] Отримання екземпляра для мови: {language}...")
    
    def _get_paddle_language(self, language: str) -> str:
        """Отримання коду мови для PaddleOCR"""
        lang_map = {
            'eng': 'en',
            'ukr': 'uk',
            LANG_ENG_UKR: 'en',
            LANG_UKR_ENG: 'en'
        }
        return lang_map.get(language, 'en')
    
    def _prepare_image_for_paddleocr(self, image: np.ndarray) -> np.ndarray:
        """Підготовка зображення для PaddleOCR"""
        import cv2
        
        logger.info(f"[PaddleOCR] Підготовка зображення: початковий формат - shape={image.shape}, dtype={image.dtype}")
        
        if image.dtype != np.uint8:
            if image.dtype in (np.float32, np.float64):
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            logger.info("[PaddleOCR] Конвертовано grayscale -> BGR")
        elif len(image.shape) == 3:
            image = self._convert_3channel_to_bgr(image)
        
        image = np.clip(image, 0, 255).astype(np.uint8)
        logger.info(f"[PaddleOCR] Фінальний формат зображення: shape={image.shape}, dtype={image.dtype}")
        return image
    
    def _convert_3channel_to_bgr(self, image: np.ndarray) -> np.ndarray:
        """Конвертація 3-канального зображення в BGR"""
        import cv2
        
        if image.shape[2] == 1:
            image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
            logger.info("[PaddleOCR] Конвертовано 1 канал -> BGR")
        elif image.shape[2] == 3:
            logger.info("[PaddleOCR] Зображення вже має 3 канали (BGR)")
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            logger.info("[PaddleOCR] Конвертовано RGBA -> BGR")
        
        return image
    
    def _perform_paddleocr_recognition(self, ocr_instance, image: np.ndarray, recognize_start: float):
        """Виконання OCR розпізнавання через PaddleOCR"""
        import time
        
        ocr_start = time.time()
        print("[PaddleOCR] Виклик методу розпізнавання...")
        logger.info("[PaddleOCR] Виклик методу розпізнавання...")
        
        try:
            if hasattr(ocr_instance, 'ocr'):
                return self._try_ocr_method(ocr_instance, image, ocr_start)
            elif hasattr(ocr_instance, 'predict'):
                return self._try_predict_method(ocr_instance, image, ocr_start)
            else:
                raise RuntimeError("PaddleOCR не має методів ocr() або predict()")
        except Exception as e:
            ocr_time = time.time() - ocr_start
            logger.error(f"[PaddleOCR] ✗ Помилка виклику (через {ocr_time:.2f} сек): {e}")
            import traceback
            logger.error(f"[PaddleOCR] Traceback: {traceback.format_exc()}")
            raise
    
    def _try_ocr_method(self, ocr_instance, image: np.ndarray, ocr_start: float):
        """Спроба використати метод ocr()"""
        import time
        
        print("[PaddleOCR] Використовується стандартний метод ocr()")
        logger.info("[PaddleOCR] Використовується стандартний метод ocr()")
        
        try:
            results = ocr_instance.ocr(image, cls=False)
            ocr_time = time.time() - ocr_start
            print(f"[PaddleOCR] ✓ Метод ocr() виконано за {ocr_time:.2f} сек")
            logger.info(f"[PaddleOCR] ✓ Метод ocr() виконано за {ocr_time:.2f} сек")
            return results
        except TypeError:
            print("[PaddleOCR] cls параметр не підтримується, викликаємо без нього")
            logger.info("[PaddleOCR] cls параметр не підтримується, викликаємо без нього")
            results = ocr_instance.ocr(image)
            ocr_time = time.time() - ocr_start
            print(f"[PaddleOCR] ✓ Метод ocr() виконано за {ocr_time:.2f} сек")
            logger.info(f"[PaddleOCR] ✓ Метод ocr() виконано за {ocr_time:.2f} сек")
            return results
    
    def _try_predict_method(self, ocr_instance, image: np.ndarray, ocr_start: float):
        """Спроба використати метод predict()"""
        import time
        
        print("[PaddleOCR] ocr() недоступний, використовується метод predict()")
        logger.warning("[PaddleOCR] ocr() недоступний, використовується метод predict()")
        
        try:
            results = ocr_instance.predict(
                image,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_thresh=0.2,
                text_det_box_thresh=0.3,
                text_rec_score_thresh=0.1
            )
            ocr_time = time.time() - ocr_start
            print(f"[PaddleOCR] ✓ Метод predict() виконано за {ocr_time:.2f} сек")
            logger.info(f"[PaddleOCR] ✓ Метод predict() виконано за {ocr_time:.2f} сек")
            return results
        except Exception as predict_error:
            print(f"[PaddleOCR] Помилка predict() з параметрами: {predict_error}, пробуємо без параметрів")
            logger.warning(f"[PaddleOCR] Помилка predict() з параметрами: {predict_error}, пробуємо без параметрів")
            try:
                results = ocr_instance.predict(image)
                ocr_time = time.time() - ocr_start
                print(f"[PaddleOCR] ✓ Метод predict() без параметрів виконано за {ocr_time:.2f} сек")
                logger.info("[PaddleOCR] ✓ Метод predict() без параметрів виконано за %.2f сек", ocr_time)
                return results
            except Exception as e2:
                ocr_time = time.time() - ocr_start
                logger.error("[PaddleOCR] ✗ Помилка predict() без параметрів: %s", e2)
                raise
    
    def _try_fallback_for_none_result(self, ocr_instance, image: np.ndarray):
        """Спроба fallback при None результаті"""
        import time
        
        warning_msg = "[PaddleOCR] ⚠️ ocr() повернув None"
        print(warning_msg)
        logger.warning(warning_msg)
        
        if not hasattr(ocr_instance, 'predict'):
            return None
        
        print("[PaddleOCR] Спроба використати predict() як fallback...")
        logger.info("[PaddleOCR] Спроба використати predict() як fallback...")
        
        try:
            fallback_start = time.time()
            results = ocr_instance.predict(image)
            fallback_time = time.time() - fallback_start
            print(f"[PaddleOCR] ✓ predict() fallback виконано за {fallback_time:.2f} сек")
            logger.info(f"[PaddleOCR] ✓ predict() fallback виконано за {fallback_time:.2f} сек")
            
            if results is None:
                print("[PaddleOCR] ⚠️ predict() також повернув None")
                logger.warning("[PaddleOCR] ⚠️ predict() також повернув None")
                return None
            
            return results
        except Exception as fallback_error:
            print(f"[PaddleOCR] ✗ predict() fallback не вдався: {fallback_error}")
            logger.warning(f"[PaddleOCR] ✗ predict() fallback не вдався: {fallback_error}")
            import traceback
            logger.error(f"[PaddleOCR] Traceback: {traceback.format_exc()}")
            return None
    
    def _try_fallback_for_empty_result(self, ocr_instance, image: np.ndarray):
        """Спроба fallback при порожньому результаті"""
        import time
        
        warning_msg = "[PaddleOCR] ⚠️ ocr() повернув порожній список - спробуємо predict() як fallback"
        print(warning_msg, flush=True)
        logger.warning(warning_msg)
        
        if not hasattr(ocr_instance, 'predict'):
            return None
        
        print("[PaddleOCR] Спроба використати predict() як fallback для порожнього списку...", flush=True)
        logger.info("[PaddleOCR] Спроба використати predict() як fallback для порожнього списку...")
        
        try:
            fallback_start = time.time()
            results = ocr_instance.predict(image)
            fallback_time = time.time() - fallback_start
            print(f"[PaddleOCR] ✓ predict() fallback виконано за {fallback_time:.2f} сек", flush=True)
            logger.info(f"[PaddleOCR] ✓ predict() fallback виконано за {fallback_time:.2f} сек")
            
            if results is None or (isinstance(results, (list, tuple)) and len(results) == 0):
                print("[PaddleOCR] ⚠️ predict() також повернув порожній результат", flush=True)
                logger.warning("[PaddleOCR] ⚠️ predict() також повернув порожній результат")
                return None
            
            print("[PaddleOCR] predict() дав результат, продовжуємо обробку...", flush=True)
            logger.info("[PaddleOCR] predict() дав результат, продовжуємо обробку...")
            return results
        except Exception as fallback_error:
            print(f"[PaddleOCR] ✗ predict() fallback не вдався: {fallback_error}", flush=True)
            logger.warning(f"[PaddleOCR] ✗ predict() fallback не вдався: {fallback_error}")
            import traceback
            logger.error(f"[PaddleOCR] Traceback: {traceback.format_exc()}")
            return None
    
    def _extract_text_from_results(self, results) -> str:
        """Витягнення тексту з результатів PaddleOCR"""
        print("[PaddleOCR] ===== ОБРОБКА РЕЗУЛЬТАТУ =====", flush=True)
        logger.info("[PaddleOCR] ===== ОБРОБКА РЕЗУЛЬТАТУ =====")
        self._log_result_structure(results)
        
        if isinstance(results, list) and len(results) > 0:
            return self._extract_text_from_list_results(results)
        
        if not isinstance(results, (list, tuple, str)) and hasattr(results, 'rec_texts'):
            return self._extract_text_from_ocrresult(results)
        
        if isinstance(results, str):
            return results.strip()
        
        return self._extract_text_from_unknown_format(results)
    
    def _log_result_structure(self, results) -> None:
        """Логування структури результату"""
        print(f"[PaddleOCR] Тип результату: {type(results)}", flush=True)
        logger.info(f"[PaddleOCR] Тип результату: {type(results)}")
        
        if results is not None:
            if isinstance(results, (list, tuple)):
                print(f"[PaddleOCR] Результат: список/кортеж, довжина={len(results)}", flush=True)
                logger.info(f"[PaddleOCR] Результат: список/кортеж, довжина={len(results)}")
                if len(results) > 0:
                    print(f"[PaddleOCR] Перший елемент: {type(results[0])}", flush=True)
                    logger.info(f"[PaddleOCR] Перший елемент: {type(results[0])}")
            else:
                print(f"[PaddleOCR] Результат: {str(results)[:200]}...", flush=True)
                logger.info(f"[PaddleOCR] Результат: {str(results)[:200]}...")
        else:
            print("[PaddleOCR] Результат: None", flush=True)
            logger.info("[PaddleOCR] Результат: None")
    
    def _extract_text_from_list_results(self, results: list) -> str:
        """Витягнення тексту зі списку результатів"""
        first_item = results[0]
        print(f"[PaddleOCR] Перший елемент: тип={type(first_item)}")
        logger.info(f"[PaddleOCR] Перший елемент: тип={type(first_item)}")
        
        if isinstance(first_item, dict):
            return self._extract_text_from_dict_result(first_item)
        
        if isinstance(first_item, (list, tuple)) and len(first_item) >= 2:
            return self._extract_text_from_standard_format(results)
        
        return self._extract_text_from_unknown_first_item(first_item)
    
    def _extract_text_from_dict_result(self, first_item: dict) -> str:
        """Витягнення тексту зі словника (новий формат predict())"""
        print(f"[PaddleOCR] Ключі в словнику: {list(first_item.keys())}")
        logger.info(f"[PaddleOCR] Ключі в словнику: {list(first_item.keys())}")
        
        if 'rec_texts' not in first_item:
            print("[PaddleOCR] ⚠️ Ключ 'rec_texts' не знайдено в словнику")
            logger.warning("[PaddleOCR] ⚠️ Ключ 'rec_texts' не знайдено в словнику")
            return ""
        
        rec_texts = first_item['rec_texts']
        rec_scores = first_item.get('rec_scores', [])
        print(f"[PaddleOCR] rec_texts знайдено: {rec_texts}")
        print(f"[PaddleOCR] rec_scores: {rec_scores}")
        logger.info(f"[PaddleOCR] rec_texts знайдено: {rec_texts}")
        logger.info(f"[PaddleOCR] rec_scores: {rec_scores}")
        
        if isinstance(rec_texts, str):
            text = rec_texts.strip()
            print(f"[PaddleOCR] Отримано рядок: '{text}'")
            logger.info(f"[PaddleOCR] Отримано рядок: '{text}'")
            return text
        
        if not isinstance(rec_texts, list):
            print("[PaddleOCR] ⚠️ rec_texts порожній або невірного типу")
            logger.warning("[PaddleOCR] ⚠️ rec_texts порожній або невірного типу")
            return ""
        
        filtered_texts = []
        for idx, t in enumerate(rec_texts):
            if not t or not str(t).strip():
                continue
            
            if idx < len(rec_scores):
                score = rec_scores[idx]
                filtered_texts.append(str(t).strip())
                print(f"[PaddleOCR] Додано текст '{t}' з впевненістю {score:.3f}")
                logger.info("[PaddleOCR] Додано текст '%s' з впевненістю %.3f", t, score)
            else:
                filtered_texts.append(str(t).strip())
                print(f"[PaddleOCR] Додано текст '{t}' (без перевірки впевненості)")
                logger.info("[PaddleOCR] Додано текст '%s' (без перевірки впевненості)", t)
        
        print(f"[PaddleOCR] Відфільтровано текстів: {len(filtered_texts)} з {len(rec_texts)}")
        logger.info(f"[PaddleOCR] Відфільтровано текстів: {len(filtered_texts)} з {len(rec_texts)}")
        
        if filtered_texts:
            text = ' '.join(filtered_texts)
            print(f"[PaddleOCR] ✓ Отримано текст: '{text}'")
            logger.info(f"[PaddleOCR] ✓ Отримано текст: '{text}'")
            return text
        
        print("[PaddleOCR] ⚠️ Всі тексти порожні або з низькою впевненістю після фільтрації")
        logger.warning("[PaddleOCR] ⚠️ Всі тексти порожні або з низькою впевненістю після фільтрації")
        return ""
    
    def _extract_text_from_standard_format(self, results: list) -> str:
        """Витягнення тексту зі стандартного формату PaddleOCR"""
        print("[PaddleOCR] Стандартний формат PaddleOCR: список списків")
        logger.info("[PaddleOCR] Стандартний формат PaddleOCR: список списків")
        texts = []
        
        for idx, result in enumerate(results):
            if not isinstance(result, (list, tuple)) or len(result) < 2:
                continue
            
            bbox = result[0]
            text_data = result[1]
            
            text_item = self._extract_text_from_text_data(text_data)
            if text_item and text_item != 'None':
                texts.append(text_item)
                print(f"[PaddleOCR] Додано текст: '{text_item}'")
                logger.info(f"[PaddleOCR] Додано текст: '{text_item}'")
        
        if texts:
            text = ' '.join(texts)
            print(f"[PaddleOCR] ✓ Отримано {len(texts)} рядків: '{text}'")
            logger.info(f"[PaddleOCR] ✓ Отримано {len(texts)} рядків: '{text}'")
            return text
        
        print("[PaddleOCR] ⚠️ Не знайдено тексту в результатах після парсингу")
        logger.warning("[PaddleOCR] ⚠️ Не знайдено тексту в результатах після парсингу")
        return ""
    
    def _extract_text_from_text_data(self, text_data) -> str:
        """Витягнення тексту з text_data"""
        if isinstance(text_data, (list, tuple)) and len(text_data) >= 1:
            return str(text_data[0]).strip()
        if isinstance(text_data, str):
            return text_data.strip()
        return str(text_data).strip()
    
    def _extract_text_from_unknown_first_item(self, first_item) -> str:
        """Витягнення тексту з невідомого формату першого елемента"""
        print(f"[PaddleOCR] ⚠️ Невідомий формат першого елемента: {type(first_item)}")
        logger.warning(f"[PaddleOCR] ⚠️ Невідомий формат першого елемента: {type(first_item)}")
        
        if isinstance(first_item, str):
            text = first_item.strip()
            print(f"[PaddleOCR] Оброблено як рядок: '{text}'")
            logger.info(f"[PaddleOCR] Оброблено як рядок: '{text}'")
            return text
        
        return ""
    
    def _extract_text_from_ocrresult(self, results) -> str:
        """Витягнення тексту з OCRResult об'єкта"""
        print("[PaddleOCR] Формат: OCRResult об'єкт")
        logger.info("[PaddleOCR] Формат: OCRResult об'єкт")
        
        try:
            rec_texts = getattr(results, 'rec_texts', None)
            if rec_texts:
                filtered_texts = [t for t in rec_texts if t and str(t).strip()]
                if filtered_texts:
                    text = ' '.join(filtered_texts)
                    print(f"[PaddleOCR] Отримано {len(filtered_texts)} рядків з OCRResult: '{text}'")
                    logger.info(f"[PaddleOCR] Отримано {len(filtered_texts)} рядків з OCRResult: '{text}'")
                    return text
        except AttributeError:
            print("[PaddleOCR] ⚠️ Помилка доступу до rec_texts")
            logger.warning("[PaddleOCR] ⚠️ Помилка доступу до rec_texts")
        
        return ""
    
    def _extract_text_from_unknown_format(self, results) -> str:
        """Витягнення тексту з невідомого формату"""
        print(f"[PaddleOCR] ⚠️ Невідомий формат результату: {type(results)}")
        logger.warning(f"[PaddleOCR] ⚠️ Невідомий формат результату: {type(results)}")
        
        try:
            text = str(results).strip()
            if text and text != 'None' and len(text) > 0:
                print(f"[PaddleOCR] Конвертовано в рядок: '{text[:200]}...'")
                logger.info(f"[PaddleOCR] Конвертовано в рядок: '{text[:200]}...'")
                return text
        except Exception as e:
            print(f"[PaddleOCR] ✗ Не вдалося конвертувати результат: {e}")
            logger.error(f"[PaddleOCR] ✗ Не вдалося конвертувати результат: {e}")
        
        if hasattr(results, '__dict__'):
            return self._extract_text_from_dict_attributes(results)
        
        return ""
    
    def _extract_text_from_dict_attributes(self, results) -> str:
        """Витягнення тексту через атрибути об'єкта"""
        try:
            for attr_name in dir(results):
                if 'text' in attr_name.lower() and not attr_name.startswith('_'):
                    try:
                        attr_value = getattr(results, attr_name)
                        if isinstance(attr_value, str) and attr_value.strip():
                            text = attr_value.strip()
                            print(f"[PaddleOCR] Знайдено текст в атрибуті {attr_name}: '{text[:200]}...'")
                            logger.info(f"[PaddleOCR] Знайдено текст в атрибуті {attr_name}: '{text[:200]}...'")
                            return text
                    except Exception:
                        pass
        except Exception:
            pass
        
        return ""
    
    def _log_recognition_complete(self, result_text: str, results, total_time: float) -> None:
        """Логування завершення розпізнавання"""
        from pathlib import Path
        
        try:
            debug_file = Path(__file__).parent.parent.parent / "paddleocr_debug.log"
            with open(debug_file, "a", encoding="utf-8") as f:
                f.write(f"[PaddleOCR] ✓ Розпізнавання завершено за {total_time:.2f} сек\n")
                f.write(f"[PaddleOCR] Результат: {len(result_text)} символів\n")
                if result_text:
                    f.write(f"[PaddleOCR] Перші 100 символів: {result_text[:100]}...\n")
                else:
                    f.write("[PaddleOCR] ⚠️ ФІНАЛЬНИЙ РЕЗУЛЬТАТ ПОРОЖНІЙ!\n")
                    f.write(f"[PaddleOCR] Тип оригінального результату: {type(results)}\n")
                    if results is not None:
                        f.write(f"[PaddleOCR] Результат (repr): {repr(results)[:500]}...\n")
                f.write(f"{'='*80}\n")
                f.flush()
        except Exception:
            pass
        
        logger.info(f"[PaddleOCR] ✓ Розпізнавання завершено за {total_time:.2f} сек")
        logger.info(f"[PaddleOCR] Результат: {len(result_text)} символів")
        
        if result_text:
            logger.info(f"[PaddleOCR] Перші 100 символів: {result_text[:100]}...")
            print(f"[PaddleOCR] ✓ Фінальний текст: '{result_text[:100]}...'", flush=True)
        else:
            logger.warning("[PaddleOCR] ⚠️ ФІНАЛЬНИЙ РЕЗУЛЬТАТ ПОРОЖНІЙ!")
            print("[PaddleOCR] ⚠️ ФІНАЛЬНИЙ РЕЗУЛЬТАТ ПОРОЖНІЙ!", flush=True)
        
        logger.info("[PaddleOCR] ===== ЗАВЕРШЕНО розпізнавання =====")

