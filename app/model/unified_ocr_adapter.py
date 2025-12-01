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
    
    def _get_strategy(self, engine: str) -> Optional[OCRStrategy]:
        """Отримання стратегії (з глобальним кешуванням)"""
        engine = engine.lower()
        
        # Спочатку перевіряємо глобальний кеш
        with _STRATEGIES_LOCK:
            if engine in _GLOBAL_STRATEGIES:
                strategy = _GLOBAL_STRATEGIES[engine]
                if strategy is not None:
                    # Додаємо до локального кешу для швидкого доступу
                    self._strategies[engine] = strategy
                    logger.info(f"[OCR] ✓ Використовується стратегія {engine} з глобального кешу (моделі вже в пам'яті)")
                    return strategy
        
        # Якщо немає в глобальному кеші, перевіряємо локальний
        if engine not in self._strategies:
            try:
                logger.info(f"[OCR] Створення стратегії для рушія: {engine}")
                import time
                strategy_start = time.time()
                
                if engine == 'tesseract':
                    strategy = TesseractStrategy()
                elif engine == 'easyocr':
                    strategy = EasyOCRStrategy()
                elif engine == 'paddleocr':
                    logger.info(f"[OCR] Створення PaddleOCRStrategy...")
                    try:
                        strategy = PaddleOCRStrategy()
                        strategy_elapsed = time.time() - strategy_start
                        logger.info(f"[OCR] PaddleOCRStrategy створено за {strategy_elapsed:.2f} секунд")
                    except Exception as e:
                        strategy_elapsed = time.time() - strategy_start
                        logger.error(f"[OCR] Помилка створення PaddleOCRStrategy (через {strategy_elapsed:.2f} сек): {e}")
                        import traceback
                        logger.error(f"[OCR] Traceback: {traceback.format_exc()}")
                        raise
                else:
                    logger.warning(f"Невідомий рушій: {engine}")
                    return None
                
                strategy_elapsed = time.time() - strategy_start
                logger.info(f"[OCR] Перевірка доступності стратегії {engine}...")
                try:
                    is_available = strategy.is_available()
                    logger.info(f"[OCR] Стратегія {engine} доступна: {is_available}")
                except Exception as e:
                    logger.error(f"[OCR] Помилка перевірки доступності стратегії {engine}: {e}")
                    is_available = False
                
                if is_available:
                    # Додаємо до глобального кешу (щоб зберегти моделі в пам'яті)
                    with _STRATEGIES_LOCK:
                        _GLOBAL_STRATEGIES[engine] = strategy
                    # Додаємо до локального кешу
                    self._strategies[engine] = strategy
                    logger.info(f"[OCR] Стратегія {engine} додана до глобального та локального кешу")
                else:
                    logger.warning(f"[OCR] Стратегія {engine} не доступна")
                    with _STRATEGIES_LOCK:
                        _GLOBAL_STRATEGIES[engine] = None
                    return None
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
                    logger.info(f"[OCR] Успішно перемкнуто на Tesseract")
                    return True
                else:
                    logger.error(f"[OCR] Не вдалося перемкнутися на Tesseract")
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
        # ВИМКНЕНО: Автоматичне перемикання для української мови
        # Дозволяємо використовувати PaddleOCR для української, щоб перевірити, чи він працює
        # Якщо потрібно, можна увімкнути це перемикання назад
        # if language == 'ukr' and self.engine_name == 'paddleocr':
        #     logger.warning(f"[OCR] ⚠️ PaddleOCR має обмежену підтримку української мови")
        #     logger.warning(f"[OCR] ⚠️ Автоматичне перемикання на EasyOCR для кращого розпізнавання кирилиці")
        #     # Спробуємо перемкнутися на EasyOCR
        #     if self._set_strategy('easyocr'):
        #         logger.info(f"[OCR] ✓ Перемкнуто на EasyOCR для української мови")
        #     elif self._set_strategy('tesseract'):
        #         logger.info(f"[OCR] ✓ Перемкнуто на Tesseract для української мови (EasyOCR недоступний)")
        #     else:
        #         logger.warning(f"[OCR] ⚠️ Не вдалося перемкнутися на інший рушій, використовуємо PaddleOCR")
        
        # Логуємо, який рушій буде використовуватися
        print(f"[OCR] Запитаний рушій: {self.engine_name}, Мова: {language}", flush=True)
        logger.info(f"[OCR] Запитаний рушій: {self.engine_name}, Мова: {language}")
        
        try:
            # Завантаження зображення, якщо потрібно
            import cv2
            if isinstance(image, str):
                processed_image = self.preprocessor.process_from_path(image)
            elif preprocess:
                processed_image = self.preprocessor.process(image)
            else:
                # Переконуємося, що це numpy array
                if not isinstance(image, np.ndarray):
                    image = np.array(image)
                if len(image.shape) == 3:
                    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    processed_image = image.astype(np.uint8) if image.dtype != np.uint8 else image
            
            # Перевірка стратегії
            if not self._current_strategy:
                # Остання спроба - перемкнутися на Tesseract
                logger.warning(f"[OCR] ⚠️ Поточна стратегія не встановлена, спроба перемкнутися на Tesseract")
                if self._set_strategy('tesseract'):
                    logger.info(f"[OCR] ✓ Успішно перемкнуто на Tesseract")
                else:
                    raise RuntimeError("Жоден OCR рушій не доступний")
            
            # Перевірка стратегії після можливого fallback
            if not self._current_strategy:
                raise RuntimeError("Жоден OCR рушій не доступний")
            
            # Опціональна сегментація на слова (для рукописного тексту)
            word_segments = []
            if split_into_words:
                try:
                    word_segments = WordSegmenter.segment_words(processed_image)
                    if len(word_segments) <= 1:
                        logger.info("[OCR] Сегментація повернула <=1 сегмент, використовуємо повне зображення")
                        word_segments = []
                except Exception as seg_error:
                    logger.warning(f"[OCR] Помилка сегментації слів: {seg_error}")
                    word_segments = []
            
            # Логуємо, який рушій використовується
            strategy_name = self._current_strategy.get_name()
            logger.info(f"[OCR] ===== ПОЧАТОК РОЗПІЗНАВАННЯ =====")
            logger.info(f"[OCR] Використовується рушій: {strategy_name}")
            logger.info(f"[OCR] Запитаний рушій: {self.engine_name}")
            logger.info(f"[OCR] Мова розпізнавання: {language}")
            logger.info(f"[OCR] Розмір обробленого зображення: {processed_image.shape}")
            
            # Розпізнавання
            import time
            recognize_start = time.time()
            print(f"[OCR] ===== ПОЧАТОК РОЗПІЗНАВАННЯ =====", flush=True)
            print(f"[OCR] Використовується рушій: {strategy_name}", flush=True)
            print(f"[OCR] Запитаний рушій: {self.engine_name}", flush=True)
            print(f"[OCR] Мова: {language}", flush=True)
            logger.info(f"[OCR] ===== ПОЧАТОК РОЗПІЗНАВАННЯ =====")
            logger.info(f"[OCR] Використовується рушій: {strategy_name}")
            logger.info(f"[OCR] Запитаний рушій: {self.engine_name}")
            logger.info(f"[OCR] Мова розпізнавання: {language}")
            logger.info(f"[OCR] Розмір обробленого зображення: {processed_image.shape}")
            
            try:
                if word_segments:
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
                else:
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
                            print(f"[OCR] ⚠️ РЕЗУЛЬТАТ ПОРОЖНІЙ!", flush=True)
                    except Exception as strategy_error:
                        logger.error(f"[OCR] ✗ Помилка в strategy.recognize(): {strategy_error}", exc_info=True)
                        print(f"[OCR] ✗ Помилка в strategy.recognize(): {strategy_error}", flush=True)
                        import traceback
                        traceback_str = traceback.format_exc()
                        print(f"[OCR] Traceback:\n{traceback_str}", flush=True)
                        logger.error(f"[OCR] Traceback:\n{traceback_str}")
                        raise
                recognize_elapsed = time.time() - recognize_start
                logger.info(f"[OCR] strategy.recognize() завершено за {recognize_elapsed:.2f} секунд")
                logger.info(f"[OCR] Розпізнавання завершено, довжина тексту: {len(text)} символів")
                if text:
                    logger.info(f"[OCR] Перші 100 символів результату: {text[:100]}...")
                else:
                    logger.warning(f"[OCR] ⚠️ РЕЗУЛЬТАТ ПОРОЖНІЙ від рушія '{strategy_name}'!")
                    print(f"[OCR] ⚠️ РЕЗУЛЬТАТ ПОРОЖНІЙ від рушія '{strategy_name}'!", flush=True)
                logger.info(f"[OCR] ===== ЗАВЕРШЕНО РОЗПІЗНАВАННЯ =====")
            except Exception as e:
                recognize_elapsed = time.time() - recognize_start
                error_msg = f"[OCR] Помилка в strategy.recognize() після {recognize_elapsed:.2f} секунд: {e}"
                print(error_msg)  # Додаємо print для гарантії виведення
                logger.error(error_msg)
                import traceback
                traceback_str = traceback.format_exc()
                print(f"[OCR] Traceback: {traceback_str}")  # Додаємо print для гарантії виведення
                logger.error(f"[OCR] Traceback: {traceback_str}")
                raise
            
            # LLM пост-обробка з валідацією
            if self.use_llm_correction and self.llm_processor and text:
                try:
                    lang_map = {
                        'eng': 'english',
                        'ukr': 'ukrainian',
                        'eng+ukr': 'ukrainian',
                        'ukr+eng': 'ukrainian'
                    }
                    lang_name = lang_map.get(language, 'english')
                    
                    # Використовуємо валідацію та виправлення
                    validation_result = self.llm_processor.validate_and_correct(text, lang_name)
                    
                    if validation_result['corrected'] and validation_result['is_valid']:
                        text = validation_result['corrected']
                        logger.info(f"[OCR] LLM виправлення застосовано. Впевненість: {validation_result['confidence']:.2f}")
                        if validation_result['changes']:
                            logger.info(f"[OCR] Зміни: {', '.join(validation_result['changes'])}")
                    else:
                        # Fallback на просте виправлення
                        text = LLMPostProcessor.simple_correction(text)
                        logger.warning(f"[OCR] LLM валідація не пройдена, використано просте виправлення")
                except Exception as e:
                    logger.warning(f"Помилка LLM пост-обробки: {e}")
                    # Використовуємо просте виправлення
                    text = LLMPostProcessor.simple_correction(text)
            elif text:
                # Завжди застосовуємо просте виправлення, навіть без LLM
                text = LLMPostProcessor.simple_correction(text)
            
            return text.strip() if text else ""
            
        except Exception as e:
            logger.error(f"[OCR] ✗ Помилка розпізнавання рушієм '{self.engine_name}': {e}")
            import traceback
            logger.error(f"[OCR] Traceback: {traceback.format_exc()}")
            # Graceful degradation - пробуємо інші рушії
            if self.engine_name != 'tesseract':
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
            
            # Якщо все не вдалося, повертаємо порожній рядок
            logger.error(f"[OCR] ✗ Всі спроби розпізнавання не вдалися")
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

