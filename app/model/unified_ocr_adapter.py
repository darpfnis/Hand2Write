"""
Unified OCR Adapter з Strategy pattern та LLM post-processing
handwrite2print/app/model/unified_ocr_adapter.py
"""
import logging
import threading
from typing import Optional, Dict, Any, List
import numpy as np

from .ocr_strategies import OCRStrategy, TesseractStrategy, EasyOCRStrategy, PaddleOCRStrategy
from .optimized_preprocessor import OptimizedPreprocessor
from .llm_postprocessor import LLMPostProcessor
from .word_segmenter import WordSegmenter

logger = logging.getLogger(__name__)

# Константа для маркування початку розпізнавання
OCR_START_MESSAGE = "[OCR] ===== ПОЧАТОК РОЗПІЗНАВАННЯ ====="

# Глобальний кеш стратегій - всі адаптери використовують ті самі екземпляри
# Це забезпечує збереження моделей в пам'яті між викликами
_GLOBAL_STRATEGIES: Dict[str, Optional[OCRStrategy]] = {}
_STRATEGIES_LOCK = threading.Lock()


class UnifiedOCRAdapter:
    """
    Уніфікований адаптер для OCR з підтримкою різних рушіїв
    Використовує Strategy pattern для легкого перемикання між рушіями
    """
    
    def __init__(self, 
                 engine: str = 'tesseract',
                 use_llm_correction: bool = False,
                 llm_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Ініціалізація адаптера
        
        Args:
            engine: назва рушія ('tesseract', 'easyocr', 'paddleocr')
            use_llm_correction: чи використовувати LLM для виправлення
            llm_config: конфігурація LLM
        """
        self.engine_name = engine.lower()
        self.use_llm_correction = use_llm_correction
        
        # Використовуємо глобальний кеш стратегій для збереження моделей в пам'яті
        # Ініціалізація стратегій (використовуємо глобальний кеш)
        self._strategies: Dict[str, OCRStrategy] = {}  # Локальний кеш для швидкого доступу
        self._current_strategy: Optional[OCRStrategy] = None
        
        # Ініціалізація препроцесора
        self.preprocessor = OptimizedPreprocessor()
        
        # Ініціалізація LLM пост-процесора
        self.llm_processor = None
        if use_llm_correction and llm_config:
            try:
                llm_kwargs = self._sanitize_llm_config(llm_config)
                if llm_kwargs:
                    self.llm_processor = LLMPostProcessor(**llm_kwargs)
                    self.use_llm_correction = self.llm_processor.is_available()
                else:
                    logger.warning("LLM конфігурація не містить необхідних полів, LLM вимкнено")
                    self.use_llm_correction = False
            except Exception as e:
                logger.warning(f"Не вдалося ініціалізувати LLM: {e}")
                self.use_llm_correction = False
        
        # Встановлення поточної стратегії
        self._set_strategy(self.engine_name)
    
    def _get_strategy_from_cache(self, engine: str) -> Optional[OCRStrategy]:
        """Отримання стратегії з кешу (глобального або локального)"""
        with _STRATEGIES_LOCK:
            if engine in _GLOBAL_STRATEGIES:
                strategy = _GLOBAL_STRATEGIES[engine]
                if strategy is not None:
                    self._strategies[engine] = strategy
                    logger.info(f"[OCR] ✓ Використовується стратегія {engine} з глобального кешу (моделі вже в пам'яті)")
                    return strategy
        return None
    
    def _create_strategy_instance(self, engine: str) -> Optional[OCRStrategy]:
        """Створення екземпляра стратегії"""
        logger.info(f"[OCR] Створення стратегії для рушія: {engine}")
        import time
        strategy_start = time.time()
        
        if engine == 'tesseract':
            return TesseractStrategy()
        if engine == 'easyocr':
            return EasyOCRStrategy()
        if engine == 'paddleocr':
            logger.info("[OCR] Створення PaddleOCRStrategy...")
            try:
                strategy = PaddleOCRStrategy()
                strategy_elapsed = time.time() - strategy_start
                logger.info(f"[OCR] PaddleOCRStrategy створено за {strategy_elapsed:.2f} секунд")
                return strategy
            except Exception as e:
                strategy_elapsed = time.time() - strategy_start
                logger.error(f"[OCR] Помилка створення PaddleOCRStrategy (через {strategy_elapsed:.2f} сек): {e}")
                import traceback
                logger.error(f"[OCR] Traceback: {traceback.format_exc()}")
                raise
        
        logger.warning(f"Невідомий рушій: {engine}")
        return None
    
    def _check_and_cache_strategy(self, engine: str, strategy: OCRStrategy) -> Optional[OCRStrategy]:
        """Перевірка доступності стратегії та додавання до кешу"""
        logger.info(f"[OCR] Перевірка доступності стратегії {engine}...")
        try:
            is_available = strategy.is_available()
            logger.info(f"[OCR] Стратегія {engine} доступна: {is_available}")
        except Exception as e:
            logger.error(f"[OCR] Помилка перевірки доступності стратегії {engine}: {e}")
            is_available = False
        
        if is_available:
            with _STRATEGIES_LOCK:
                _GLOBAL_STRATEGIES[engine] = strategy
            self._strategies[engine] = strategy
            logger.info(f"[OCR] Стратегія {engine} додана до глобального та локального кешу")
            return strategy
        
        logger.warning(f"[OCR] Стратегія {engine} не доступна")
        with _STRATEGIES_LOCK:
            _GLOBAL_STRATEGIES[engine] = None
        return None
    
    def _get_strategy(self, engine: str) -> Optional[OCRStrategy]:
        """Отримання стратегії (з глобальним кешуванням)"""
        engine = engine.lower()
        
        # Спочатку перевіряємо кеш
        cached_strategy = self._get_strategy_from_cache(engine)
        if cached_strategy:
            return cached_strategy
        
        # Якщо немає в кеші, створюємо нову стратегію
        if engine not in self._strategies:
            try:
                strategy = self._create_strategy_instance(engine)
                if strategy is None:
                    return None
                return self._check_and_cache_strategy(engine, strategy)
            except Exception as e:
                logger.error(f"[OCR] Помилка ініціалізації стратегії {engine}: {e}")
                import traceback
                logger.error(f"[OCR] Traceback: {traceback.format_exc()}")
                return None
        
        return self._strategies.get(engine)
    
    def _set_strategy(self, engine: str) -> bool:
        """Встановлення поточної стратегії"""
        logger.info(f"[OCR] Встановлення стратегії: {engine}")
        strategy = self._get_strategy(engine)
        if strategy:
            self._current_strategy = strategy
            self.engine_name = engine.lower()
            logger.info(f"[OCR] Стратегія '{engine}' успішно встановлена ({strategy.get_name()})")
            return True
        else:
            logger.warning(f"[OCR] Стратегія '{engine}' не доступна")
            # Автоматичний fallback на Tesseract для всіх недоступних рушіїв
            if engine != 'tesseract':
                logger.info(f"[OCR] Автоматичне перемикання на Tesseract (оригінальний рушій '{engine}' недоступний)")
                if self._set_strategy('tesseract'):
                    logger.info("[OCR] Успішно перемкнуто на Tesseract")
                    return True
                else:
                    logger.error("[OCR] Не вдалося перемкнутися на Tesseract")
            return False
    
    def get_available_engines(self) -> List[str]:
        """Отримання списку доступних рушіїв"""
        available = []
        for engine in ['tesseract', 'easyocr', 'paddleocr']:
            strategy = self._get_strategy(engine)
            if strategy and strategy.is_available():
                available.append(engine)
        return available
    
    def recognize(self, 
                  image: np.ndarray, 
                  language: str = 'eng',
                  preprocess: bool = True,
                  split_into_words: bool = False) -> str:
        """
        Розпізнавання тексту з обробкою помилок
        
        Args:
            image: зображення (numpy array або шлях до файлу)
            language: код мови ('eng', 'ukr', 'eng+ukr')
            preprocess: чи застосовувати препроцесинг
            split_into_words: розбивати зображення на слова перед OCR
            
        Returns:
            розпізнаний текст
        """
        self._log_recognition_request(language)
        
        try:
            processed_image = self._prepare_image(image, preprocess)
            self._ensure_strategy_available()
            word_segments = self._segment_words_if_needed(processed_image, split_into_words)
            
            strategy_name = self._current_strategy.get_name()
            self._log_recognition_start(strategy_name, language, processed_image)
            
            text = self._recognize_text(processed_image, language, word_segments, strategy_name)
            text = self._apply_llm_correction(text, language)
            
            return text.strip() if text else ""
            
        except Exception as e:
            return self._handle_recognition_error(e, image, language, preprocess, split_into_words)
    
    def _log_recognition_request(self, language: str) -> None:
        """Логування запиту на розпізнавання"""
        print(f"[OCR] Запитаний рушій: {self.engine_name}, Мова: {language}", flush=True)
        logger.info(f"[OCR] Запитаний рушій: {self.engine_name}, Мова: {language}")
    
    def _prepare_image(self, image, preprocess: bool) -> np.ndarray:
        """Підготовка зображення для розпізнавання"""
        import cv2
        
        if isinstance(image, str):
            return self.preprocessor.process_from_path(image)
        if preprocess:
            return self.preprocessor.process(image)
        
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.astype(np.uint8) if image.dtype != np.uint8 else image
    
    def _ensure_strategy_available(self) -> None:
        """Перевірка та встановлення стратегії OCR"""
        if not self._current_strategy:
            logger.warning("[OCR] ⚠️ Поточна стратегія не встановлена, спроба перемкнутися на Tesseract")
            if self._set_strategy('tesseract'):
                logger.info("[OCR] ✓ Успішно перемкнуто на Tesseract")
            else:
                raise RuntimeError("Жоден OCR рушій не доступний")
        
        if not self._current_strategy:
            raise RuntimeError("Жоден OCR рушій не доступний")
    
    def _segment_words_if_needed(self, processed_image: np.ndarray, split_into_words: bool) -> list:
        """Сегментація зображення на слова, якщо потрібно"""
        if not split_into_words:
            return []
        
        try:
            word_segments = WordSegmenter.segment_words(processed_image)
            if len(word_segments) <= 1:
                logger.info("[OCR] Сегментація повернула <=1 сегмент, використовуємо повне зображення")
                return []
            return word_segments
        except Exception as seg_error:
            logger.warning(f"[OCR] Помилка сегментації слів: {seg_error}")
            return []
    
    def _log_recognition_start(self, strategy_name: str, language: str, processed_image: np.ndarray) -> None:
        """Логування початку розпізнавання"""
        logger.info(OCR_START_MESSAGE)
        logger.info(f"[OCR] Використовується рушій: {strategy_name}")
        logger.info(f"[OCR] Запитаний рушій: {self.engine_name}")
        logger.info(f"[OCR] Мова розпізнавання: {language}")
        logger.info(f"[OCR] Розмір обробленого зображення: {processed_image.shape}")
        
        print(OCR_START_MESSAGE, flush=True)
        print(f"[OCR] Використовується рушій: {strategy_name}", flush=True)
        print(f"[OCR] Запитаний рушій: {self.engine_name}", flush=True)
        print(f"[OCR] Мова: {language}", flush=True)
    
    def _recognize_text(self, processed_image: np.ndarray, language: str, 
                       word_segments: list, strategy_name: str) -> str:
        """Виконання розпізнавання тексту"""
        import time
        recognize_start = time.time()
        
        try:
            if word_segments:
                text = self._recognize_word_segments(word_segments, language)
            else:
                text = self._recognize_full_image(processed_image, language, strategy_name)
            
            recognize_elapsed = time.time() - recognize_start
            self._log_recognition_result(text, strategy_name, recognize_elapsed)
            return text
        except Exception as e:
            recognize_elapsed = time.time() - recognize_start
            self._log_recognition_error(e, recognize_elapsed)
            raise
    
    def _recognize_word_segments(self, word_segments: list, language: str) -> str:
        """Розпізнавання окремих сегментів слів"""
        print(f"[OCR] Розпізнавання {len(word_segments)} сегментів...", flush=True)
        recognized_words = []
        
        for idx, segment in enumerate(word_segments, start=1):
            logger.info(f"[OCR] Розпізнавання сегмента #{idx} (bbox={segment.bbox})")
            print(f"[OCR] Розпізнавання сегмента #{idx}...", flush=True)
            
            try:
                segment_text = self._current_strategy.recognize(segment.image, language)
                if segment_text:
                    recognized_words.append(segment_text.strip())
                    print(f"[OCR] Сегмент #{idx}: '{segment_text[:50]}...'", flush=True)
                else:
                    logger.warning(f"[OCR] Порожній результат для сегмента #{idx}")
                    print(f"[OCR] ⚠️ Сегмент #{idx}: порожній результат", flush=True)
            except Exception as seg_error:
                logger.error(f"[OCR] Помилка розпізнавання сегмента #{idx}: {seg_error}")
                print(f"[OCR] ✗ Помилка сегмента #{idx}: {seg_error}", flush=True)
        
        text = " ".join(recognized_words).strip()
        print(f"[OCR] Об'єднано {len(recognized_words)} сегментів: {len(text)} символів", flush=True)
        return text
    
    def _recognize_full_image(self, processed_image: np.ndarray, language: str, strategy_name: str) -> str:
        """Розпізнавання повного зображення"""
        print(f"[OCR] Виклик strategy.recognize() для рушія '{strategy_name}'...", flush=True)
        logger.info(f"[OCR] Виклик strategy.recognize() для рушія '{strategy_name}'...")
        
        try:
            text = self._current_strategy.recognize(processed_image, language)
            result_len = len(text) if text else 0
            logger.info(f"[OCR] strategy.recognize() повернув результат: {result_len} символів")
            print(f"[OCR] strategy.recognize() повернув результат: {result_len} символів", flush=True)
            
            if text:
                print(f"[OCR] Перші 100 символів: {text[:100]}...", flush=True)
            else:
                print("[OCR] ⚠️ РЕЗУЛЬТАТ ПОРОЖНІЙ!", flush=True)
            
            return text
        except Exception as strategy_error:
            logger.error(f"[OCR] ✗ Помилка в strategy.recognize(): {strategy_error}", exc_info=True)
            print(f"[OCR] ✗ Помилка в strategy.recognize(): {strategy_error}", flush=True)
            import traceback
            traceback_str = traceback.format_exc()
            print(f"[OCR] Traceback:\n{traceback_str}", flush=True)
            logger.error(f"[OCR] Traceback:\n{traceback_str}")
            raise
    
    def _log_recognition_result(self, text: str, strategy_name: str, elapsed: float) -> None:
        """Логування результату розпізнавання"""
        logger.info(f"[OCR] strategy.recognize() завершено за {elapsed:.2f} секунд")
        logger.info(f"[OCR] Розпізнавання завершено, довжина тексту: {len(text)} символів")
        
        if text:
            logger.info(f"[OCR] Перші 100 символів результату: {text[:100]}...")
        else:
            logger.warning(f"[OCR] ⚠️ РЕЗУЛЬТАТ ПОРОЖНІЙ від рушія '{strategy_name}'!")
            print(f"[OCR] ⚠️ РЕЗУЛЬТАТ ПОРОЖНІЙ від рушія '{strategy_name}'!", flush=True)
        
        logger.info("[OCR] ===== ЗАВЕРШЕНО РОЗПІЗНАВАННЯ =====")
    
    def _log_recognition_error(self, error: Exception, elapsed: float) -> None:
        """Логування помилки розпізнавання"""
        error_msg = f"[OCR] Помилка в strategy.recognize() після {elapsed:.2f} секунд: {error}"
        print(error_msg)
        logger.error(error_msg)
        import traceback
        traceback_str = traceback.format_exc()
        print(f"[OCR] Traceback: {traceback_str}")
        logger.error(f"[OCR] Traceback: {traceback_str}")
    
    def _apply_llm_correction(self, text: str, language: str) -> str:
        """Застосування LLM корекції до тексту"""
        if self.use_llm_correction and self.llm_processor and text:
            return self._apply_llm_validation(text, language)
        if text:
            return LLMPostProcessor.simple_correction(text)
        return text
    
    def _apply_llm_validation(self, text: str, language: str) -> str:
        """Застосування LLM валідації та виправлення"""
        try:
            lang_map = {
                'eng': 'english',
                'ukr': 'ukrainian',
                'eng+ukr': 'ukrainian',
                'ukr+eng': 'ukrainian'
            }
            lang_name = lang_map.get(language, 'english')
            
            validation_result = self.llm_processor.validate_and_correct(text, lang_name)
            
            if validation_result['corrected'] and validation_result['is_valid']:
                text = validation_result['corrected']
                logger.info(f"[OCR] LLM виправлення застосовано. Впевненість: {validation_result['confidence']:.2f}")
                if validation_result['changes']:
                    logger.info(f"[OCR] Зміни: {', '.join(validation_result['changes'])}")
                return text
            
            text = LLMPostProcessor.simple_correction(text)
            logger.warning("[OCR] LLM валідація не пройдена, використано просте виправлення")
            return text
        except Exception as e:
            logger.warning(f"Помилка LLM пост-обробки: {e}")
            return LLMPostProcessor.simple_correction(text)
    
    def _handle_recognition_error(self, error: Exception, image, language: str, 
                                  preprocess: bool, split_into_words: bool) -> str:
        """Обробка помилки розпізнавання з fallback на Tesseract"""
        logger.error(f"[OCR] ✗ Помилка розпізнавання рушієм '{self.engine_name}': {error}")
        import traceback
        logger.error(f"[OCR] Traceback: {traceback.format_exc()}")
        
        if self.engine_name != 'tesseract':
            return self._try_fallback_to_tesseract(image, language, preprocess, split_into_words)
        
        logger.error("[OCR] ✗ Всі спроби розпізнавання не вдалися")
        return ""
    
    def _try_fallback_to_tesseract(self, image, language: str, preprocess: bool, 
                                   split_into_words: bool) -> str:
        """Спроба fallback на Tesseract"""
        logger.warning(f"[OCR] ⚠️ СПРОБА FALLBACK НА TESSERACT (оригінальний рушій: {self.engine_name})")
        try:
            old_engine = self.engine_name
            if self._set_strategy('tesseract'):
                logger.warning(f"[OCR] ⚠️ FALLBACK: Використовується Tesseract замість {old_engine}")
                fallback_result = self.recognize(image, language, preprocess, split_into_words=split_into_words)
                logger.warning(f"[OCR] ⚠️ FALLBACK: Результат від Tesseract: {len(fallback_result)} символів")
                return fallback_result
        except Exception as fallback_error:
            logger.error(f"[OCR] ✗ Fallback на Tesseract також не вдався: {fallback_error}")
        
        logger.error("[OCR] ✗ Всі спроби розпізнавання не вдалися")
        return ""
    
    def switch_engine(self, engine: str) -> bool:
        """
        Перемикання рушія
        
        Args:
            engine: назва рушія
            
        Returns:
            True якщо успішно перемкнуто
        """
        return self._set_strategy(engine)
    
    def get_current_engine(self) -> Optional[str]:
        """Отримання назви поточного рушія"""
        return self._current_strategy.get_name() if self._current_strategy else None
    
    def set_preprocessing_config(self, config: Dict[str, Any]):
        """Встановлення конфігурації препроцесингу"""
        self.preprocessor.config.update(config)
    
    def enable_llm_correction(self, llm_config: Optional[Dict[str, Any]] = None):
        """Увімкнення LLM корекції"""
        try:
            if llm_config:
                llm_kwargs = self._sanitize_llm_config(llm_config)
                if llm_kwargs:
                    self.llm_processor = LLMPostProcessor(**llm_kwargs)
                else:
                    logger.warning("LLM конфігурація пуста, неможливо увімкнути LLM")
                    self.use_llm_correction = False
                    return
            elif not self.llm_processor:
                # Створюємо з дефолтними налаштуваннями
                self.llm_processor = LLMPostProcessor(api_type="local")
            
            self.use_llm_correction = self.llm_processor.is_available()
        except Exception as e:
            logger.warning(f"Не вдалося увімкнути LLM: {e}")
            self.use_llm_correction = False
    
    def disable_llm_correction(self):
        """Вимкнення LLM корекції"""
        self.llm_processor = None
        self.use_llm_correction = False

    @staticmethod
    def _sanitize_llm_config(llm_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Повертає лише дозволені ключі для LLMPostProcessor"""
        if not llm_config:
            return {}
        allowed_keys = ('api_type', 'api_key', 'api_url')
        return {
            key: llm_config.get(key)
            for key in allowed_keys
            if llm_config.get(key)
        }

