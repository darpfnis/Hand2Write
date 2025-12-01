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
                is_available, version, _ = check_pytorch_availability()
                
                if not is_available:
                    raise RuntimeError(f"PyTorch не доступний: {error}")
                
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
            
            # Підготовка зображення
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            # Обробка зображення
            # EasyOCR працює краще з оригінальним зображенням або з мінімальною обробкою
            # Використовуємо мінімальну обробку для всіх мов
            logger.info(f"[EasyOCR] Використання мінімальної обробки для мови: {language}")
            # Просто переконуємося, що зображення в правильному форматі (RGB, 3 канали)
            if len(image.shape) == 2:
                # Grayscale -> RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                logger.info("[EasyOCR] Конвертовано grayscale -> RGB")
            elif len(image.shape) == 3:
                if image.shape[2] == 1:
                    # 1 канал -> RGB
                    image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
                    logger.info("[EasyOCR] Конвертовано 1 канал -> RGB")
                elif image.shape[2] == 3:
                    # Вже 3 канали, переконуємося що це RGB (EasyOCR очікує RGB)
                    # Якщо це BGR, конвертуємо в RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    logger.info("[EasyOCR] Конвертовано BGR -> RGB")
                elif image.shape[2] == 4:
                    # RGBA -> RGB
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                    logger.info("[EasyOCR] Конвертовано RGBA -> RGB")
            logger.info("[EasyOCR] Використовується оригінальне зображення з мінімальною обробкою")
            
            # Мапінг мов
            # Для української використовуємо тільки 'uk' для кращого розпізнавання кирилиці
            lang_map = {
                'eng': ['en'],
                'ukr': ['uk'],  # Тільки українська для кращого розпізнавання
                'eng+ukr': ['en', 'uk'],
                'ukr+eng': ['uk', 'en']
            }
            easyocr_langs = lang_map.get(language, ['en'])
            
            logger.info(f"[EasyOCR] Використовується мова: {easyocr_langs}")
            
            # Розпізнавання з оптимізованими параметрами для рукописного тексту
            # Зменшуємо пороги для кращого розпізнавання рукопису
            width_ths = 0.5  # Зменшено для рукописного тексту
            height_ths = 0.5  # Зменшено для рукописного тексту
            paragraph = False
            
            logger.info(f"[EasyOCR] Параметри: width_ths={width_ths}, height_ths={height_ths}, мова={language}")
            
            result = reader.readtext(
                image,
                paragraph=paragraph,
                detail=1,
                width_ths=width_ths,
                height_ths=height_ths,
                allowlist=None,  # Дозволяємо всі символи
                blocklist=None   # Не блокуємо символи
            )
            
            logger.info(f"[EasyOCR] Отримано {len(result)} результатів")
            if result:
                logger.info(f"[EasyOCR] Перші 3 результати: {result[:3]}")
            
            # Фільтрація та сортування
            # Для рукописного тексту використовуємо дуже низький поріг
            confidence_threshold = 0.05  # Знижено для рукописного тексту
            filtered_result = []
            for idx, detection in enumerate(result):
                if isinstance(detection, (list, tuple)) and len(detection) >= 3:
                    bbox, txt, conf = detection[0], detection[1], detection[2]
                    logger.info(f"[EasyOCR] Результат {idx}: txt='{txt}' (тип: {type(txt)}), conf={conf} (тип: {type(conf)})")
                    
                    # Перевіряємо, чи текст не порожній
                    if not txt or (isinstance(txt, str) and not txt.strip()):
                        logger.warning(f"[EasyOCR] ✗ Пропущено (порожній текст): '{txt}'")
                        continue
                    
                    # Перевіряємо впевненість
                    if isinstance(conf, (int, float)):
                        if conf > confidence_threshold:
                            # Додаємо результат
                            filtered_result.append((bbox, txt, conf))
                            if conf < 0.3:
                                logger.warning(f"[EasyOCR] ⚠️ Додано результат з низькою впевненістю: '{txt}' (conf={conf:.3f}) - можлива помилка розпізнавання")
                            else:
                                logger.info(f"[EasyOCR] ✓ Додано: '{txt}' (conf={conf:.3f})")
                        else:
                            logger.info(f"[EasyOCR] ✗ Пропущено (низька впевненість {conf:.3f} <= {confidence_threshold}): '{txt}'")
                    else:
                        # Якщо впевненість не число, все одно додаємо (може бути None або інший тип)
                        logger.warning(f"[EasyOCR] ⚠️ Впевненість не число ({type(conf)}), але додаємо результат: '{txt}'")
                        filtered_result.append((bbox, txt, conf))
                else:
                    logger.warning(f"[EasyOCR] Невірний формат результату {idx}: {detection} (тип: {type(detection)}, довжина: {len(detection) if hasattr(detection, '__len__') else 'N/A'})")
            
            logger.info(f"[EasyOCR] Відфільтровано {len(filtered_result)} результатів з {len(result)}")
            
            # Сортування за позицією
            if filtered_result:
                filtered_result.sort(key=lambda x: (
                    x[0][0][1] if isinstance(x[0], (list, tuple)) and len(x[0]) > 0 else 0,
                    x[0][0][0] if isinstance(x[0], (list, tuple)) and len(x[0]) > 0 else 0
                ))
                
                # Обчислюємо середню впевненість для оцінки якості
                confidences = [conf for _, _, conf in filtered_result if isinstance(conf, (int, float))]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    min_confidence = min(confidences)
                    logger.info(f"[EasyOCR] Якість розпізнавання: середня впевненість={avg_confidence:.3f}, мінімальна={min_confidence:.3f}")
                    if min_confidence < 0.3:
                        logger.warning("[EasyOCR] ⚠️ Увага: є результати з дуже низькою впевненістю (< 0.3) - можливі помилки розпізнавання")
            
            text = ' '.join([txt for _, txt, _ in filtered_result])
            return text.strip()
            
        except Exception as e:
            logger.error(f"Помилка EasyOCR: {e}")
            raise


class PaddleOCRStrategy(OCRStrategy):
    """Стратегія для PaddleOCR"""
    
    def __init__(self):
        self._instances = {}  # Кеш екземплярів по мовах
        self._available = False
        self._creating_instances = {}  # Захист від одночасного створення
        self._init()
    
    def _init(self):
        """Лінива ініціалізація PaddleOCR - тільки перевірка доступності, без створення моделей"""
        global _ENGINE_AVAILABILITY_CACHE
        
        # Перевірка кешу
        cache_key = 'paddleocr'
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
            pytorch_available, pytorch_version, _ = check_pytorch_availability()
            
            if not pytorch_available:
                # Логуємо тільки один раз, зменшуємо рівень до INFO
                if cache_key not in _ENGINE_AVAILABILITY_CACHE:
                    # Спрощене повідомлення - детальна інформація в FINAL_SOLUTION_PYTORCH.md
                    logger.info("PaddleOCR не доступний через проблеми з PyTorch DLL. Використовується Tesseract.")
                self._available = False
                _ENGINE_AVAILABILITY_CACHE[cache_key] = False
                return
            
            # Перевірка PaddlePaddle (тільки імпорт, без створення моделей)
            try:
                # Тільки перевіряємо, чи можна імпортувати paddle
                # НЕ створюємо PaddleOCR екземпляр тут, щоб не блокувати програму
                logger.info("[PaddleOCR] Спробуємо імпортувати paddle...")
                import time
                import_start = time.time()
                import paddle
                import_elapsed = time.time() - import_start
                logger.info(f"[PaddleOCR] paddle успішно імпортовано за {import_elapsed:.2f} секунд")
                
                # НЕ імпортуємо paddleocr тут, бо це може викликати створення моделей
                # Перевірку paddleocr зробимо тільки при створенні екземпляра
                logger.info("[PaddleOCR] paddle доступний, paddleocr буде перевірено при створенні екземпляра")
                
                self._available = True
                _ENGINE_AVAILABILITY_CACHE[cache_key] = True
                logger.info(f"[PaddleOCR] PaddleOCR доступний (PyTorch {pytorch_version})")
            except ImportError as e:
                if cache_key not in _ENGINE_AVAILABILITY_CACHE:
                    logger.warning(f"PaddleOCR не доступний: PaddlePaddle не встановлено: {e}")
                self._available = False
                _ENGINE_AVAILABILITY_CACHE[cache_key] = False
            except Exception as e:
                if cache_key not in _ENGINE_AVAILABILITY_CACHE:
                    logger.warning(f"PaddleOCR не доступний: {e}")
                self._available = False
                _ENGINE_AVAILABILITY_CACHE[cache_key] = False
                
        except ImportError:
            # Якщо pytorch_helper недоступний, використовуємо базову перевірку
            try:
                import torch
                # НЕ імпортуємо paddle тут, бо це викликає створення моделей
                # Просто перевіряємо PyTorch
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
        # Мапінг мов для PaddleOCR
        # PaddleOCR v3.3.2 підтримує:
        # - 'en' - англійська
        # - 'ch' - китайська
        # - 'ru' - російська (кирилиця)
        # - 'uk' - українська (якщо доступно, інакше використовуємо 'ru' для кирилиці)
        # - 'korean', 'japan', 'french', 'german' тощо
        # Для української спробуємо 'uk', якщо не працює - fallback на 'ru'
        lang_map = {
            'eng': 'en',
            'ukr': 'uk',  # Спробуємо українську, якщо не підтримується - fallback на 'ru'
            LANG_ENG_UKR: 'en',
            LANG_UKR_ENG: 'en'
        }
        paddle_lang = lang_map.get(language, 'en')
        
            # Перевіряємо кеш
        if paddle_lang in self._instances:
            logger.info(f"[PaddleOCR] ✓ Використовується закешований екземпляр для мови: {paddle_lang}")
            return self._instances[paddle_lang]
        
        # Перевіряємо, чи не створюється вже екземпляр для цієї мови
        if paddle_lang in self._creating_instances:
            # Чекаємо, поки інший потік створить екземпляр
            import time
            max_wait = 60  # Максимум 1 хвилина
            wait_time = 0
            while paddle_lang in self._creating_instances and wait_time < max_wait:
                time.sleep(0.5)
                wait_time += 0.5
                if paddle_lang in self._instances:
                    return self._instances[paddle_lang]
            
            if paddle_lang not in self._instances:
                raise RuntimeError(f"Не вдалося створити екземпляр PaddleOCR для мови {paddle_lang}")
            return self._instances[paddle_lang]
        
        # Створюємо новий екземпляр для цієї мови
        self._creating_instances[paddle_lang] = True
        try:
            import time
            
            # Перевіряємо, яка версія PaddleOCR використовується
            logger.info("[PaddleOCR] Перевірка доступності paddle та paddleocr...")
            try:
                # Спочатку перевіряємо paddle (це може зайняти час, але необхідно)
                import paddle
                logger.info("[PaddleOCR] paddle успішно імпортовано")
            except ImportError as e:
                logger.error(f"[PaddleOCR] paddle не встановлено: {e}")
                raise RuntimeError(f"PaddlePaddle не встановлено: {e}")
            except Exception as e:
                logger.warning(f"[PaddleOCR] Помилка імпорту paddle: {e}")
                # Продовжуємо, можливо це не критично
            
            try:
                import paddleocr
                # Визначаємо, звідки імпортується PaddleOCR
                paddleocr_file = getattr(paddleocr, '__file__', None)
                if paddleocr_file:
                    paddleocr_path = Path(paddleocr_file).resolve()
                    # Перевіряємо, чи це локальна версія
                    current_file = Path(__file__).resolve()
                    base_dir_check = current_file.parent.parent.parent
                    local_paddleocr_dir = base_dir_check / "resources" / "models" / "v3.3.2 source code" / "PaddlePaddle-PaddleOCR-95dc316"
                    if str(local_paddleocr_dir.absolute()) in str(paddleocr_path):
                        logger.info(f"[PaddleOCR] ✓ Використовується ЛОКАЛЬНА версія PaddleOCR з: {paddleocr_path}")
                    else:
                        logger.info(f"[PaddleOCR] Використовується pip-встановлена версія PaddleOCR з: {paddleocr_path}")
                if hasattr(paddleocr, '__version__'):
                    logger.info(f"[PaddleOCR] Версія PaddleOCR: {paddleocr.__version__}")
            except Exception as e:
                logger.error(f"[PaddleOCR] Не вдалося імпортувати paddleocr: {e}")
                raise
            
            from paddleocr import PaddleOCR
            
            # Логуємо налаштування перед створенням
            logger.info(f"[PaddleOCR] ===== ПОЧАТОК створення екземпляра для мови: {paddle_lang} =====")
            logger.info(f"[PaddleOCR] PADDLE_PDX_CACHE_HOME: {os.environ.get('PADDLE_PDX_CACHE_HOME', 'не встановлено')}")
            logger.info(f"[PaddleOCR] PADDLEX_HOME: {os.environ.get('PADDLEX_HOME', 'не встановлено')}")
            start_time = time.time()
            
            # PaddleOCR v3.3.2 не підтримує use_gpu параметр
            # Використовуємо тільки підтримувані параметри
            try:
                logger.info(f"[PaddleOCR] Виклик PaddleOCR(lang='{paddle_lang}', use_angle_cls=False)...")
                logger.info("[PaddleOCR] Це може зайняти кілька хвилин при першому запуску...")
                instance = PaddleOCR(lang=paddle_lang, use_angle_cls=False)
                elapsed_time = time.time() - start_time
                logger.info(f"[PaddleOCR] ✓ Екземпляр успішно створено за {elapsed_time:.2f} секунд ({elapsed_time/60:.1f} хвилин)")
            except Exception as init_error:
                # Якщо 'uk' не підтримується, спробуємо 'ru' для кирилиці
                if paddle_lang == 'uk':
                    logger.warning(f"[PaddleOCR] Мова 'uk' не підтримується, спробуємо 'ru' для кирилиці: {init_error}")
                    try:
                        paddle_lang = 'ru'
                        logger.info(f"[PaddleOCR] Виклик PaddleOCR(lang='{paddle_lang}', use_angle_cls=False)...")
                        instance = PaddleOCR(lang=paddle_lang, use_angle_cls=False)
                        elapsed_time = time.time() - start_time
                        logger.info(f"[PaddleOCR] ✓ Екземпляр успішно створено з 'ru' за {elapsed_time:.2f} секунд")
                    except Exception as fallback_error:
                        elapsed_time = time.time() - start_time
                        logger.error(f"[PaddleOCR] ✗ Помилка при створенні екземпляра з 'ru' (через {elapsed_time:.2f} сек): {fallback_error}")
                        import traceback
                        logger.error(f"[PaddleOCR] Traceback: {traceback.format_exc()}")
                        raise
                else:
                    elapsed_time = time.time() - start_time
                    logger.error(f"[PaddleOCR] ✗ Помилка при створенні екземпляра (через {elapsed_time:.2f} сек): {init_error}")
                import traceback
                logger.error(f"[PaddleOCR] Traceback: {traceback.format_exc()}")
                raise
            
            # Перевіряємо, чи екземпляр працює
            try:
                logger.info("[PaddleOCR] Перевірка екземпляра...")
                logger.info(f"[PaddleOCR] Тип: {type(instance).__name__}")
                logger.info(f"[PaddleOCR] Має метод ocr: {hasattr(instance, 'ocr')}")
                logger.info(f"[PaddleOCR] Має метод predict: {hasattr(instance, 'predict')}")
            except Exception as e:
                logger.warning(f"[PaddleOCR] Не вдалося перевірити екземпляр: {e}")
            
            # Кешуємо екземпляр
            self._instances[paddle_lang] = instance
            logger.info(f"[PaddleOCR] ✓ Екземпляр закешовано для мови: {paddle_lang}")
            logger.info(f"[PaddleOCR] ===== ЗАВЕРШЕНО створення екземпляра для мови: {paddle_lang} =====")
            return instance
        except Exception as e:
            logger.error(f"[PaddleOCR] Помилка ініціалізації для мови {paddle_lang}: {e}")
            import traceback
            logger.error(f"[PaddleOCR] Traceback: {traceback.format_exc()}")
            raise
        finally:
            # Видаляємо зі списку створюваних
            if paddle_lang in self._creating_instances:
                del self._creating_instances[paddle_lang]
    
    def is_available(self) -> bool:
        return self._available
    
    def get_name(self) -> str:
        return "PaddleOCR"
    
    def recognize(self, image: np.ndarray, language: str) -> str:
        """Розпізнавання через PaddleOCR"""
        # КРИТИЧНО: Логуємо в файл на самому початку, щоб переконатися, що метод викликається
        import os
        from pathlib import Path
        debug_file = Path(__file__).parent.parent.parent / "paddleocr_debug.log"
        try:
            with open(debug_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"[PaddleOCR] ===== ВИКЛИК recognize() =====\n")
                f.write(f"[PaddleOCR] Час: {__import__('datetime').datetime.now()}\n")
                f.write(f"[PaddleOCR] self._available = {self._available}\n")
                f.write(f"[PaddleOCR] Мова: {language}\n")
                f.write(f"[PaddleOCR] Розмір зображення: {image.shape if hasattr(image, 'shape') else 'N/A'}\n")
                f.flush()
        except Exception as log_error:
            pass
        
        # Також використовуємо звичайне логування
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
        
        if not self._available:
            error_msg = "PaddleOCR не доступний!"
            print(f"[PaddleOCR] ✗ {error_msg}")
            logger.error(f"[PaddleOCR] ✗ {error_msg}")
            raise RuntimeError(error_msg)
        
        try:
            import cv2
            import time
            
            print("[PaddleOCR] ===== ПОЧАТОК розпізнавання =====")
            logger.info("[PaddleOCR] ===== ПОЧАТОК розпізнавання =====")
            print(f"[PaddleOCR] Мова: {language}, розмір зображення: {image.shape}")
            logger.info(f"[PaddleOCR] Мова: {language}, розмір зображення: {image.shape}")
            print(f"[PaddleOCR] Тип зображення: {type(image)}, dtype: {image.dtype if hasattr(image, 'dtype') else 'N/A'}")
            logger.info(f"[PaddleOCR] Тип зображення: {type(image)}, dtype: {image.dtype if hasattr(image, 'dtype') else 'N/A'}")
            
            recognize_start = time.time()
            print(f"[PaddleOCR] Отримання екземпляра для мови: {language}...")
            ocr_instance = self._get_instance(language)
            print("[PaddleOCR] Екземпляр отримано")
            get_instance_time = time.time() - recognize_start
            logger.info(f"[PaddleOCR] Екземпляр отримано за {get_instance_time:.2f} сек")
            
            # Логуємо, який екземпляр використовується
            # Мапінг має відповідати тому, що використовується в _get_instance()
            lang_map = {
                'eng': 'en',
                'ukr': 'uk',  # Спробуємо українську, fallback на 'ru' в _get_instance()
                'eng+ukr': 'en',
                'ukr+eng': 'en'
            }
            paddle_lang = lang_map.get(language, 'en')
            logger.info(f"[PaddleOCR] Використовується PaddleOCR екземпляр для мови: {paddle_lang} (запитана мова: {language})")
            
            # Підготовка зображення для PaddleOCR
            # PaddleOCR працює краще з BGR зображеннями (3 канали)
            logger.info(f"[PaddleOCR] Підготовка зображення: початковий формат - shape={image.shape}, dtype={image.dtype}")
            
            if image.dtype != np.uint8:
                if image.dtype in (np.float32, np.float64):
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Конвертуємо в BGR формат (3 канали), якщо потрібно
            if len(image.shape) == 2:
                # Grayscale -> BGR
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                logger.info("[PaddleOCR] Конвертовано grayscale -> BGR")
            elif len(image.shape) == 3:
                if image.shape[2] == 1:
                    # 1 канал -> BGR
                    image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
                    logger.info("[PaddleOCR] Конвертовано 1 канал -> BGR")
                elif image.shape[2] == 3:
                    # Вже 3 канали, переконуємося що це BGR (не RGB)
                    # PaddleOCR очікує BGR формат
                    logger.info("[PaddleOCR] Зображення вже має 3 канали (BGR)")
                elif image.shape[2] == 4:
                    # RGBA -> BGR
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                    logger.info("[PaddleOCR] Конвертовано RGBA -> BGR")
            
            # Переконуємося, що значення в правильному діапазоні
            image = np.clip(image, 0, 255).astype(np.uint8)
            logger.info(f"[PaddleOCR] Фінальний формат зображення: shape={image.shape}, dtype={image.dtype}")
            
            # Розпізнавання
            # Використовуємо стандартний метод ocr() як основний, оскільки він надійніший
            # Метод predict() може мати інший формат результатів
            results = None
            ocr_start = time.time()
            try:
                print("[PaddleOCR] Виклик методу розпізнавання...")
                logger.info("[PaddleOCR] Виклик методу розпізнавання...")
                
                # Спочатку пробуємо стандартний метод ocr() - він завжди повертає однаковий формат
                if hasattr(ocr_instance, 'ocr'):
                    print("[PaddleOCR] Використовується стандартний метод ocr()")
                    logger.info("[PaddleOCR] Використовується стандартний метод ocr()")
                    try:
                        results = ocr_instance.ocr(image, cls=False)
                        ocr_time = time.time() - ocr_start
                        print(f"[PaddleOCR] ✓ Метод ocr() виконано за {ocr_time:.2f} сек")
                        logger.info(f"[PaddleOCR] ✓ Метод ocr() виконано за {ocr_time:.2f} сек")
                    except TypeError:
                        # Якщо cls параметр не підтримується, викликаємо без нього
                        print("[PaddleOCR] cls параметр не підтримується, викликаємо без нього")
                        logger.info("[PaddleOCR] cls параметр не підтримується, викликаємо без нього")
                        results = ocr_instance.ocr(image)
                        ocr_time = time.time() - ocr_start
                        print(f"[PaddleOCR] ✓ Метод ocr() виконано за {ocr_time:.2f} сек")
                        logger.info(f"[PaddleOCR] ✓ Метод ocr() виконано за {ocr_time:.2f} сек")
                # Якщо ocr() недоступний, пробуємо predict()
                elif hasattr(ocr_instance, 'predict'):
                    print("[PaddleOCR] ocr() недоступний, використовується метод predict()")
                    logger.warning("[PaddleOCR] ocr() недоступний, використовується метод predict()")
                    try:
                        # Використовуємо predict() з параметрами для кращого розпізнавання рукописного тексту
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
                    except Exception as predict_error:
                        print(f"[PaddleOCR] Помилка predict() з параметрами: {predict_error}, пробуємо без параметрів")
                        logger.warning(f"[PaddleOCR] Помилка predict() з параметрами: {predict_error}, пробуємо без параметрів")
                        try:
                            results = ocr_instance.predict(image)
                            ocr_time = time.time() - ocr_start
                            print(f"[PaddleOCR] ✓ Метод predict() без параметрів виконано за {ocr_time:.2f} сек")
                            logger.info(f"[PaddleOCR] ✓ Метод predict() без параметрів виконано за {ocr_time:.2f} сек")
                        except Exception as e2:
                            ocr_time = time.time() - ocr_start
                            logger.error(f"[PaddleOCR] ✗ Помилка predict() без параметрів: {e2}")
                            raise
                else:
                    raise RuntimeError("PaddleOCR не має методів ocr() або predict()")
            except Exception as e:
                ocr_time = time.time() - ocr_start
                logger.error(f"[PaddleOCR] ✗ Помилка виклику (через {ocr_time:.2f} сек): {e}")
                import traceback
                logger.error(f"[PaddleOCR] Traceback: {traceback.format_exc()}")
                raise
            
            # Обробка результату
            print("[PaddleOCR] ===== ОБРОБКА РЕЗУЛЬТАТУ =====", flush=True)
            logger.info("[PaddleOCR] ===== ОБРОБКА РЕЗУЛЬТАТУ =====")
            print(f"[PaddleOCR] Тип результату: {type(results)}", flush=True)
            logger.info(f"[PaddleOCR] Тип результату: {type(results)}")
            # Не виводимо весь результат, бо він може бути дуже великим
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
            print(f"[PaddleOCR] Результат is None: {results is None}", flush=True)
            logger.info(f"[PaddleOCR] Результат is None: {results is None}")
            if results is not None:
                print(f"[PaddleOCR] Результат bool(): {bool(results)}")
                logger.info(f"[PaddleOCR] Результат bool(): {bool(results)}")
                if isinstance(results, (list, tuple)):
                    print(f"[PaddleOCR] Довжина результату (якщо список): {len(results)}")
                    logger.info(f"[PaddleOCR] Довжина результату (якщо список): {len(results)}")
                if hasattr(results, '__len__'):
                    try:
                        print(f"[PaddleOCR] len(result): {len(results)}")
                        logger.info(f"[PaddleOCR] len(result): {len(results)}")
                    except:
                        pass
            
            # Перевіряємо порожні результати (None, [], тощо)
            # Якщо ocr() повернув порожній результат, спробуємо predict() як fallback
            if results is None:
                warning_msg = f"[PaddleOCR] ⚠️ ocr() повернув None"
                print(warning_msg)
                logger.warning(warning_msg)
                
                # Спробуємо predict() як fallback, якщо доступний
                if hasattr(ocr_instance, 'predict'):
                    print("[PaddleOCR] Спроба використати predict() як fallback...")
                    logger.info("[PaddleOCR] Спроба використати predict() як fallback...")
                    try:
                        fallback_start = time.time()
                        results = ocr_instance.predict(image)
                        fallback_time = time.time() - fallback_start
                        print(f"[PaddleOCR] ✓ predict() fallback виконано за {fallback_time:.2f} сек")
                        logger.info(f"[PaddleOCR] ✓ predict() fallback виконано за {fallback_time:.2f} сек")
                        
                        # Перевіряємо, чи fallback дав результат
                        if results is None:
                            print("[PaddleOCR] ⚠️ predict() також повернув None")
                            logger.warning("[PaddleOCR] ⚠️ predict() також повернув None")
                            return ""
                    except Exception as fallback_error:
                        print(f"[PaddleOCR] ✗ predict() fallback не вдався: {fallback_error}")
                        logger.warning(f"[PaddleOCR] ✗ predict() fallback не вдався: {fallback_error}")
                        import traceback
                        logger.error(f"[PaddleOCR] Traceback: {traceback.format_exc()}")
                        return ""
                else:
                    # Якщо predict() недоступний, повертаємо порожній результат
                    return ""
            
            # Перевіряємо порожній список (може бути валидним результатом, якщо текст не знайдено)
            # АЛЕ спробуємо predict() як fallback, бо ocr() може не знайти текст, а predict() може знайти
            if isinstance(results, (list, tuple)) and len(results) == 0:
                warning_msg = f"[PaddleOCR] ⚠️ ocr() повернув порожній список - спробуємо predict() як fallback"
                print(warning_msg, flush=True)
                logger.warning(warning_msg)
                
                # Спробуємо predict() як fallback
                if hasattr(ocr_instance, 'predict'):
                    print("[PaddleOCR] Спроба використати predict() як fallback для порожнього списку...", flush=True)
                    logger.info("[PaddleOCR] Спроба використати predict() як fallback для порожнього списку...")
                    try:
                        fallback_start = time.time()
                        results = ocr_instance.predict(image)
                        fallback_time = time.time() - fallback_start
                        print(f"[PaddleOCR] ✓ predict() fallback виконано за {fallback_time:.2f} сек", flush=True)
                        logger.info(f"[PaddleOCR] ✓ predict() fallback виконано за {fallback_time:.2f} сек")
                        
                        # Перевіряємо, чи fallback дав результат
                        if results is None or (isinstance(results, (list, tuple)) and len(results) == 0):
                            print("[PaddleOCR] ⚠️ predict() також повернув порожній результат", flush=True)
                            logger.warning("[PaddleOCR] ⚠️ predict() також повернув порожній результат")
                            return ""
                        # Якщо predict() дав результат, продовжуємо обробку
                        print("[PaddleOCR] predict() дав результат, продовжуємо обробку...", flush=True)
                        logger.info("[PaddleOCR] predict() дав результат, продовжуємо обробку...")
                    except Exception as fallback_error:
                        print(f"[PaddleOCR] ✗ predict() fallback не вдався: {fallback_error}", flush=True)
                        logger.warning(f"[PaddleOCR] ✗ predict() fallback не вдався: {fallback_error}")
                        import traceback
                        logger.error(f"[PaddleOCR] Traceback: {traceback.format_exc()}")
                        return ""
                else:
                    # Якщо predict() недоступний, повертаємо порожній результат
                    print(f"[PaddleOCR] predict() недоступний, повертаємо порожній результат", flush=True)
                    return ""
            
            # PaddleOCR може повертати різні формати залежно від версії та методу
            text = ""
            
            # Детальне логування структури результату
            print("[PaddleOCR] Детальна інформація про результат:")
            logger.info("[PaddleOCR] Детальна інформація про результат:")
            print(f"  - Тип: {type(results)}")
            logger.info(f"  - Тип: {type(results)}")
            if isinstance(results, (list, tuple)):
                print(f"  - Довжина: {len(results)}")
                logger.info(f"  - Довжина: {len(results)}")
                if len(results) > 0:
                    print(f"  - Перший елемент тип: {type(results[0])}")
                    logger.info(f"  - Перший елемент тип: {type(results[0])}")
                    if isinstance(results[0], (list, tuple)):
                        print(f"  - Перший елемент довжина: {len(results[0])}")
                        logger.info(f"  - Перший елемент довжина: {len(results[0])}")
                        if len(results[0]) > 0:
                            print(f"  - Перший елемент[0] тип: {type(results[0][0])}")
                            logger.info(f"  - Перший елемент[0] тип: {type(results[0][0])}")
                            if len(results[0]) > 1:
                                print(f"  - Перший елемент[1] тип: {type(results[0][1])}")
                                logger.info(f"  - Перший елемент[1] тип: {type(results[0][1])}")
                                print(f"  - Перший елемент[1] значення: {results[0][1]}")
                                logger.info(f"  - Перший елемент[1] значення: {results[0][1]}")
            
            # Спочатку перевіряємо, чи це список
            if isinstance(results, list) and len(results) > 0:
                print(f"[PaddleOCR] Формат: список, довжина: {len(results)}")
                logger.info(f"[PaddleOCR] Формат: список, довжина: {len(results)}")
                
                # Беремо перший елемент
                first_item = results[0]
                print(f"[PaddleOCR] Перший елемент: тип={type(first_item)}")
                logger.info(f"[PaddleOCR] Перший елемент: тип={type(first_item)}")
                
                # Перевіряємо, чи це словник з rec_texts (новий формат predict())
                if isinstance(first_item, dict):
                    print(f"[PaddleOCR] Ключі в словнику: {list(first_item.keys())}")
                    logger.info(f"[PaddleOCR] Ключі в словнику: {list(first_item.keys())}")
                    
                    if 'rec_texts' in first_item:
                        rec_texts = first_item['rec_texts']
                        rec_scores = first_item.get('rec_scores', [])
                        print(f"[PaddleOCR] rec_texts знайдено: {rec_texts}")
                        print(f"[PaddleOCR] rec_scores: {rec_scores}")
                        logger.info(f"[PaddleOCR] rec_texts знайдено: {rec_texts}")
                        logger.info(f"[PaddleOCR] rec_scores: {rec_scores}")
                        
                        if rec_texts and isinstance(rec_texts, list):
                            # Фільтруємо порожні рядки та низьку впевненість
                            filtered_texts = []
                            for idx, t in enumerate(rec_texts):
                                # Перевіряємо, чи текст не порожній
                                if t and str(t).strip():
                                    # Перевіряємо впевненість (якщо є)
                                    if idx < len(rec_scores):
                                        score = rec_scores[idx]
                                        # ТИМЧАСОВО: прибираємо фільтрацію за впевненістю для діагностики
                                        # Додаємо всі тексти незалежно від впевненості
                                        filtered_texts.append(str(t).strip())
                                        print(f"[PaddleOCR] Додано текст '{t}' з впевненістю {score:.3f}")
                                        logger.info(f"[PaddleOCR] Додано текст '{t}' з впевненістю {score:.3f}")
                                    else:
                                        # Якщо немає scores, додаємо всі не порожні тексти
                                        filtered_texts.append(str(t).strip())
                                        print(f"[PaddleOCR] Додано текст '{t}' (без перевірки впевненості)")
                                        logger.info(f"[PaddleOCR] Додано текст '{t}' (без перевірки впевненості)")
                            
                            print(f"[PaddleOCR] Відфільтровано текстів: {len(filtered_texts)} з {len(rec_texts)}")
                            logger.info(f"[PaddleOCR] Відфільтровано текстів: {len(filtered_texts)} з {len(rec_texts)}")
                            
                            if filtered_texts:
                                text = ' '.join(filtered_texts)
                                print(f"[PaddleOCR] ✓ Отримано текст: '{text}'")
                                logger.info(f"[PaddleOCR] ✓ Отримано текст: '{text}'")
                            else:
                                print(f"[PaddleOCR] ⚠️ Всі тексти порожні або з низькою впевненістю після фільтрації")
                                logger.warning(f"[PaddleOCR] ⚠️ Всі тексти порожні або з низькою впевненістю після фільтрації")
                        elif rec_texts and isinstance(rec_texts, str):
                            text = rec_texts.strip()
                            print(f"[PaddleOCR] Отримано рядок: '{text}'")
                            logger.info(f"[PaddleOCR] Отримано рядок: '{text}'")
                        else:
                            print(f"[PaddleOCR] ⚠️ rec_texts порожній або невірного типу")
                            logger.warning(f"[PaddleOCR] ⚠️ rec_texts порожній або невірного типу")
                    else:
                        print(f"[PaddleOCR] ⚠️ Ключ 'rec_texts' не знайдено в словнику")
                        logger.warning(f"[PaddleOCR] ⚠️ Ключ 'rec_texts' не знайдено в словнику")
                
                # Стандартний формат PaddleOCR ocr(): [[[bbox], (text, confidence)], ...]
                elif isinstance(first_item, (list, tuple)) and len(first_item) >= 2:
                    print(f"[PaddleOCR] Стандартний формат PaddleOCR: список списків")
                    logger.info(f"[PaddleOCR] Стандартний формат PaddleOCR: список списків")
                    texts = []
                    for idx, result in enumerate(results):
                        print(f"[PaddleOCR] Елемент {idx}: тип={type(result)}, довжина={len(result) if isinstance(result, (list, tuple)) else 'N/A'}")
                        logger.info(f"[PaddleOCR] Елемент {idx}: тип={type(result)}, довжина={len(result) if isinstance(result, (list, tuple)) else 'N/A'}")
                        
                        if isinstance(result, (list, tuple)) and len(result) >= 2:
                            # Формат: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence)]
                            bbox = result[0]
                            text_data = result[1]
                            
                            print(f"[PaddleOCR] Елемент {idx}: bbox тип={type(bbox)}, text_data тип={type(text_data)}")
                            logger.info(f"[PaddleOCR] Елемент {idx}: bbox тип={type(bbox)}, text_data тип={type(text_data)}")
                            
                            # text_data може бути (text, confidence) або просто text
                            if isinstance(text_data, (list, tuple)) and len(text_data) >= 1:
                                text_item = str(text_data[0]).strip()
                                confidence = text_data[1] if len(text_data) > 1 else None
                                print(f"[PaddleOCR] text_data як список/кортеж: text='{text_item}', conf={confidence}")
                                logger.info(f"[PaddleOCR] text_data як список/кортеж: text='{text_item}', conf={confidence}")
                            elif isinstance(text_data, str):
                                text_item = text_data.strip()
                                confidence = None
                                print(f"[PaddleOCR] text_data як рядок: '{text_item}'")
                                logger.info(f"[PaddleOCR] text_data як рядок: '{text_item}'")
                            else:
                                # Спробуємо конвертувати в рядок
                                text_item = str(text_data).strip()
                                confidence = None
                                print(f"[PaddleOCR] text_data інший тип, конвертовано: '{text_item}'")
                                logger.info(f"[PaddleOCR] text_data інший тип, конвертовано: '{text_item}'")
                            
                            # Додаємо текст, навіть якщо він порожній (може бути пробіл)
                            if text_item and text_item != 'None':
                                texts.append(text_item)
                                conf_str = f" (conf={confidence:.3f})" if confidence is not None else ""
                                print(f"[PaddleOCR] Додано текст: '{text_item}'{conf_str}")
                                logger.info(f"[PaddleOCR] Додано текст: '{text_item}'{conf_str}")
                            else:
                                print(f"[PaddleOCR] Пропущено порожній або None текст: '{text_item}'")
                                logger.warning(f"[PaddleOCR] Пропущено порожній або None текст: '{text_item}'")
                        elif isinstance(result, (list, tuple)) and len(result) == 1:
                            # Можливо, це просто текст без bbox
                            text_item = str(result[0]).strip()
                            if text_item and text_item != 'None':
                                texts.append(text_item)
                                print(f"[PaddleOCR] Додано текст (без bbox): '{text_item}'")
                                logger.info(f"[PaddleOCR] Додано текст (без bbox): '{text_item}'")
                        else:
                            print(f"[PaddleOCR] ⚠️ Неочікуваний формат елемента: тип={type(result)}, довжина={len(result) if isinstance(result, (list, tuple)) else 'N/A'}")
                            logger.warning(f"[PaddleOCR] ⚠️ Неочікуваний формат елемента: тип={type(result)}, довжина={len(result) if isinstance(result, (list, tuple)) else 'N/A'}")
                            print(f"[PaddleOCR] Елемент: {result}")
                            logger.warning(f"[PaddleOCR] Елемент: {result}")
                    
                    if texts:
                        text = ' '.join(texts)
                        print(f"[PaddleOCR] ✓ Отримано {len(texts)} рядків: '{text}'")
                        logger.info(f"[PaddleOCR] ✓ Отримано {len(texts)} рядків: '{text}'")
                    else:
                        print(f"[PaddleOCR] ⚠️ Не знайдено тексту в результатах після парсингу")
                        logger.warning(f"[PaddleOCR] ⚠️ Не знайдено тексту в результатах після парсингу")
                        # Спробуємо вивести весь результат для діагностики
                        print(f"[PaddleOCR] Повний результат для діагностики: {results}")
                        logger.warning(f"[PaddleOCR] Повний результат для діагностики: {results}")
                else:
                    print(f"[PaddleOCR] ⚠️ Невідомий формат першого елемента: {type(first_item)}")
                    logger.warning(f"[PaddleOCR] ⚠️ Невідомий формат першого елемента: {type(first_item)}")
                    print(f"[PaddleOCR] Перший елемент: {first_item}")
                    logger.warning(f"[PaddleOCR] Перший елемент: {first_item}")
                    # Спробуємо обробити як рядок, якщо можливо
                    try:
                        if isinstance(first_item, str):
                            text = first_item.strip()
                            print(f"[PaddleOCR] Оброблено як рядок: '{text}'")
                            logger.info(f"[PaddleOCR] Оброблено як рядок: '{text}'")
                    except:
                        pass
            elif not isinstance(results, (list, tuple, str)) and hasattr(results, 'rec_texts'):
                # OCRResult об'єкт (старий формат) - перевіряємо, що це не список/кортеж/рядок
                print(f"[PaddleOCR] Формат: OCRResult об'єкт")
                logger.info(f"[PaddleOCR] Формат: OCRResult об'єкт")
                try:
                    rec_texts = getattr(results, 'rec_texts', None)
                    print(f"[PaddleOCR] results.rec_texts: {rec_texts}")
                    logger.info(f"[PaddleOCR] results.rec_texts: {rec_texts}")
                    if rec_texts:
                        filtered_texts = [t for t in rec_texts if t and str(t).strip()]
                        if filtered_texts:
                            text = ' '.join(filtered_texts)
                            print(f"[PaddleOCR] Отримано {len(filtered_texts)} рядків з OCRResult: '{text}'")
                            logger.info(f"[PaddleOCR] Отримано {len(filtered_texts)} рядків з OCRResult: '{text}'")
                        else:
                            print(f"[PaddleOCR] ⚠️ OCRResult.rec_texts порожній після фільтрації")
                            logger.warning(f"[PaddleOCR] ⚠️ OCRResult.rec_texts порожній після фільтрації")
                    else:
                        print(f"[PaddleOCR] ⚠️ OCRResult.rec_texts порожній")
                        logger.warning(f"[PaddleOCR] ⚠️ OCRResult.rec_texts порожній")
                except AttributeError:
                    print(f"[PaddleOCR] ⚠️ Помилка доступу до rec_texts")
                    logger.warning(f"[PaddleOCR] ⚠️ Помилка доступу до rec_texts")
            elif isinstance(results, str):
                # Можливо, це просто рядок
                print(f"[PaddleOCR] Формат: рядок")
                logger.info(f"[PaddleOCR] Формат: рядок")
                text = results.strip()
                print(f"[PaddleOCR] Отримано рядок: '{text}'")
                logger.info(f"[PaddleOCR] Отримано рядок: '{text}'")
            else:
                print(f"[PaddleOCR] ⚠️ Невідомий формат результату: {type(results)}")
                logger.warning(f"[PaddleOCR] ⚠️ Невідомий формат результату: {type(results)}")
                print(f"[PaddleOCR] Повний результат: {results}")
                logger.warning(f"[PaddleOCR] Повний результат: {results}")
                
                # Спробуємо різні способи витягти текст
                try:
                    # Спробуємо конвертувати в рядок
                    text = str(results).strip()
                    if text and text != 'None' and len(text) > 0:
                        print(f"[PaddleOCR] Конвертовано в рядок: '{text[:200]}...'")
                        logger.info(f"[PaddleOCR] Конвертовано в рядок: '{text[:200]}...'")
                    else:
                        text = ""
                except Exception as e:
                    print(f"[PaddleOCR] ✗ Не вдалося конвертувати результат: {e}")
                    logger.error(f"[PaddleOCR] ✗ Не вдалося конвертувати результат: {e}")
                    text = ""
                
                # Якщо не вдалося, спробуємо витягти через атрибути
                if not text:
                    try:
                        if hasattr(results, '__dict__'):
                            print(f"[PaddleOCR] Спробуємо витягти через __dict__: {results.__dict__}")
                            logger.warning(f"[PaddleOCR] Спробуємо витягти через __dict__: {results.__dict__}")
                            # Шукаємо атрибути, що можуть містити текст
                            for attr_name in dir(results):
                                if 'text' in attr_name.lower() and not attr_name.startswith('_'):
                                    try:
                                        attr_value = getattr(results, attr_name)
                                        if isinstance(attr_value, str) and attr_value.strip():
                                            text = attr_value.strip()
                                            print(f"[PaddleOCR] Знайдено текст в атрибуті {attr_name}: '{text[:200]}...'")
                                            logger.info(f"[PaddleOCR] Знайдено текст в атрибуті {attr_name}: '{text[:200]}...'")
                                            break
                                    except:
                                        pass
                    except:
                        pass
            
            result_text = text.strip() if text else ""
            total_time = time.time() - recognize_start
            
            # Записуємо результат у файл
            try:
                debug_file = Path(__file__).parent.parent.parent / "paddleocr_debug.log"
                with open(debug_file, "a", encoding="utf-8") as f:
                    f.write(f"[PaddleOCR] ✓ Розпізнавання завершено за {total_time:.2f} сек\n")
                    f.write(f"[PaddleOCR] Результат: {len(result_text)} символів\n")
                    if result_text:
                        f.write(f"[PaddleOCR] Перші 100 символів: {result_text[:100]}...\n")
                    else:
                        f.write(f"[PaddleOCR] ⚠️ ФІНАЛЬНИЙ РЕЗУЛЬТАТ ПОРОЖНІЙ!\n")
                        f.write(f"[PaddleOCR] Тип оригінального результату: {type(results)}\n")
                        if results is not None:
                            f.write(f"[PaddleOCR] Результат (repr): {repr(results)[:500]}...\n")
                    f.write(f"{'='*80}\n")
                    f.flush()
            except:
                pass
            
            logger.info(f"[PaddleOCR] ✓ Розпізнавання завершено за {total_time:.2f} сек")
            logger.info(f"[PaddleOCR] Результат: {len(result_text)} символів")
            if result_text:
                logger.info(f"[PaddleOCR] Перші 100 символів: {result_text[:100]}...")
                print(f"[PaddleOCR] ✓ Фінальний текст: '{result_text[:100]}...'", flush=True)
            else:
                logger.warning(f"[PaddleOCR] ⚠️ ФІНАЛЬНИЙ РЕЗУЛЬТАТ ПОРОЖНІЙ!")
                print(f"[PaddleOCR] ⚠️ ФІНАЛЬНИЙ РЕЗУЛЬТАТ ПОРОЖНІЙ!", flush=True)
                print(f"[PaddleOCR] Тип оригінального результату: {type(results)}", flush=True)
                logger.warning(f"[PaddleOCR] Тип оригінального результату: {type(results)}")
                if results is not None:
                    print(f"[PaddleOCR] Результат (repr): {repr(results)[:500]}...", flush=True)
                    logger.warning(f"[PaddleOCR] Результат (repr): {repr(results)[:500]}...")
                
                # Додаткова спроба: якщо результат не порожній, але текст не розпізнано,
                # спробуємо витягти текст іншим способом
                if results and not isinstance(results, str):
                    try:
                        # Спробуємо конвертувати весь результат в рядок
                        alternative_text = str(results).strip()
                        if alternative_text and len(alternative_text) > 10:  # Якщо є щось значуще
                            print(f"[PaddleOCR] Альтернативна спроба: '{alternative_text[:200]}...'", flush=True)
                            logger.warning(f"[PaddleOCR] Альтернативна спроба: '{alternative_text[:200]}...'")
                    except:
                        pass
            logger.info(f"[PaddleOCR] ===== ЗАВЕРШЕНО розпізнавання =====")
            
            return result_text
            
        except Exception as e:
            # КРИТИЧНО: Логуємо помилку з деталями
            import sys
            import traceback
            error_msg = f"Помилка PaddleOCR: {e}"
            print(f"[PaddleOCR] ✗ {error_msg}", flush=True)
            sys.stdout.write(f"[PaddleOCR] ✗ {error_msg}\n")
            sys.stdout.flush()
            logger.error(f"[PaddleOCR] ✗ {error_msg}", exc_info=True)
            traceback_str = traceback.format_exc()
            print(f"[PaddleOCR] Traceback:\n{traceback_str}", flush=True)
            logger.error(f"[PaddleOCR] Traceback:\n{traceback_str}")
            # Також записуємо в файл для діагностики
            try:
                with open("paddleocr_error.log", "a", encoding="utf-8") as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Помилка PaddleOCR: {e}\n")
                    f.write(f"Traceback:\n{traceback_str}\n")
                    f.write(f"{'='*80}\n")
            except:
                pass
            raise

