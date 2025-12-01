"""
Модуль для попереднього завантаження моделей OCR
handwrite2print/app/model/model_preloader.py
"""
import logging
import threading
from typing import Optional, Callable
from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class ModelPreloaderThread(QThread):
    """Потік для попереднього завантаження моделей OCR"""
    
    progress_updated = pyqtSignal(str)  # Сигнал для оновлення прогресу
    finished_signal = pyqtSignal(bool)  # Сигнал завершення (успіх/неуспіх)
    
    def __init__(self, progress_callback: Optional[Callable[[str], None]] = None):
        super().__init__()
        self.progress_callback = progress_callback
        self.easyocr_loaded = False
        self.paddleocr_loaded = False
        
    def run(self):
        """Завантаження моделей в фоновому потоці (паралельно)"""
        try:
            # Перевіряємо доступність PyTorch
            try:
                from .pytorch_helper import check_pytorch_availability
                is_available, _, _ = check_pytorch_availability()
                if not is_available:
                    logger.info("[Preloader] PyTorch недоступний, пропускаємо завантаження моделей")
                    self.finished_signal.emit(False)
                    return
            except Exception as e:
                logger.warning(f"[Preloader] Помилка перевірки PyTorch: {e}")
                self.finished_signal.emit(False)
                return
            
            # Паралельне завантаження EasyOCR та PaddleOCR
            self._update_progress("Паралельне завантаження моделей OCR...")
            
            # Результати завантаження
            easyocr_result = {'success': False, 'error': None}
            paddleocr_result = {'success': False, 'error': None}
            
            # Функції для завантаження
            def load_easyocr():
                try:
                    self._update_progress("Завантаження EasyOCR моделей...")
                    self._preload_easyocr()
                    easyocr_result['success'] = True
                    logger.info("[Preloader] ✓ EasyOCR моделі завантажено")
                except Exception as e:
                    easyocr_result['error'] = str(e)
                    logger.warning(f"[Preloader] Не вдалося завантажити EasyOCR: {e}")
            
            def load_paddleocr():
                try:
                    self._update_progress("Завантаження PaddleOCR моделей...")
                    self._preload_paddleocr()
                    paddleocr_result['success'] = True
                    logger.info("[Preloader] ✓ PaddleOCR моделі завантажено")
                except Exception as e:
                    paddleocr_result['error'] = str(e)
                    logger.warning(f"[Preloader] Не вдалося завантажити PaddleOCR: {e}")
            
            # Запускаємо обидва завантаження паралельно
            easyocr_thread = threading.Thread(target=load_easyocr, daemon=True)
            paddleocr_thread = threading.Thread(target=load_paddleocr, daemon=True)
            
            easyocr_thread.start()
            paddleocr_thread.start()
            
            # Чекаємо завершення обох потоків
            easyocr_thread.join()
            paddleocr_thread.join()
            
            self.easyocr_loaded = easyocr_result['success']
            self.paddleocr_loaded = paddleocr_result['success']
            
            if self.easyocr_loaded or self.paddleocr_loaded:
                self._update_progress("Моделі готові")
                self.finished_signal.emit(True)
            else:
                self._update_progress("Моделі не завантажено")
                self.finished_signal.emit(False)
            
        except Exception as e:
            logger.error(f"[Preloader] Помилка завантаження моделей: {e}")
            self.finished_signal.emit(False)
    
    def _update_progress(self, message: str):
        """Оновлення прогресу"""
        logger.info(f"[Preloader] {message}")
        self.progress_updated.emit(message)
        if self.progress_callback:
            try:
                self.progress_callback(message)
            except Exception:
                pass
    
    def _preload_easyocr(self):
        """Попереднє завантаження EasyOCR для популярних мов"""
        try:
            # Використовуємо глобальний кеш через UnifiedOCRAdapter
            from .unified_ocr_adapter import UnifiedOCRAdapter
            
            # Створюємо адаптер, який використовує глобальний кеш стратегій
            # Це примусово завантажує моделі в пам'ять
            self._update_progress("Ініціалізація EasyOCR...")
            adapter = UnifiedOCRAdapter(engine='easyocr')
            strategy = adapter._get_strategy('easyocr')
            
            if not strategy or not strategy.is_available():
                logger.info("[Preloader] EasyOCR недоступний, пропускаємо")
                return
            
            # Завантажуємо Reader для популярних мов
            # Це примусово викликає завантаження моделей
            popular_languages = ['ukr', 'eng']  # Українська та англійська
            
            for lang in popular_languages:
                try:
                    self._update_progress(f"Завантаження моделей EasyOCR ({lang})...")
                    # Викликаємо _get_reader для попереднього завантаження моделей
                    # Це створює easyocr.Reader, який завантажує моделі в пам'ять
                    if hasattr(strategy, '_get_reader'):
                        strategy._get_reader(lang)  # type: ignore
                        logger.info(f"[Preloader] ✓ EasyOCR Reader для '{lang}' готовий (моделі в пам'яті)")
                    else:
                        logger.warning(f"[Preloader] Стратегія {type(strategy)} не має методу _get_reader")
                except Exception as e:
                    logger.warning(f"[Preloader] Не вдалося завантажити EasyOCR для '{lang}': {e}")
                    
        except Exception as e:
            logger.warning(f"[Preloader] Помилка попереднього завантаження EasyOCR: {e}")
            raise
    
    def _preload_paddleocr(self):
        """Попереднє завантаження PaddleOCR для популярних мов"""
        try:
            # Використовуємо глобальний кеш через UnifiedOCRAdapter
            from .unified_ocr_adapter import UnifiedOCRAdapter
            
            # Створюємо адаптер, який використовує глобальний кеш стратегій
            # Це примусово завантажує моделі в пам'ять
            self._update_progress("Ініціалізація PaddleOCR...")
            adapter = UnifiedOCRAdapter(engine='paddleocr')
            strategy = adapter._get_strategy('paddleocr')
            
            if not strategy or not strategy.is_available():
                logger.info("[Preloader] PaddleOCR недоступний, пропускаємо")
                return
            
            # Завантажуємо екземпляри для популярних мов
            # Це примусово викликає створення PaddleOCR, який завантажує моделі
            popular_languages = ['ukr', 'eng']  # Українська та англійська
            
            for lang in popular_languages:
                try:
                    self._update_progress(f"Завантаження моделей PaddleOCR ({lang})...")
                    # Викликаємо _get_instance для попереднього завантаження моделей
                    # Це створює PaddleOCR екземпляр, який завантажує моделі в пам'ять
                    if hasattr(strategy, '_get_instance'):
                        strategy._get_instance(lang)  # type: ignore
                        logger.info(f"[Preloader] ✓ PaddleOCR екземпляр для '{lang}' готовий (моделі в пам'яті)")
                    else:
                        logger.warning(f"[Preloader] Стратегія {type(strategy)} не має методу _get_instance")
                except Exception as e:
                    logger.warning(f"[Preloader] Не вдалося завантажити PaddleOCR для '{lang}': {e}")
                    
        except Exception as e:
            logger.warning(f"[Preloader] Помилка попереднього завантаження PaddleOCR: {e}")
            raise

