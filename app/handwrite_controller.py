"""
Контролер - зв'язок між Model та View
handwrite2print/app/controller.py
"""
import logging
from PyQt6.QtCore import QThread, pyqtSignal
from model.unified_ocr_adapter import UnifiedOCRAdapter
from model.ocr_config import OCRConfig
from model.handwrite_export import TextExporter

logger = logging.getLogger(__name__)


class RecognitionThread(QThread):
    """Потік для розпізнавання тексту"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)  # value, message

    def __init__(self, image_path, language, ocr_engine, split_into_words=False):
        super().__init__()
        self.image_path = image_path
        self.language = language
        self.ocr_engine = ocr_engine
        self.split_into_words = split_into_words

    def run(self):
        """Виконання розпізнавання"""
        try:
            # Завантаження конфігурації
            config = OCRConfig()

            # Створення адаптера з конфігурацією
            llm_config = config.get_llm_config()
            use_llm = llm_config.get('enabled', False)

            ocr = UnifiedOCRAdapter(
                engine=self.ocr_engine,
                use_llm_correction=use_llm,
                llm_config=llm_config if use_llm else None
            )

            # Встановлення конфігурації препроцесингу
            preprocessing_config = config.get_preprocessing_config()
            ocr.set_preprocessing_config(preprocessing_config)

            # Розпізнавання (препроцесинг виконується автоматично)
            print("[Controller] ===== ПОЧАТОК РОЗПІЗНАВАННЯ =====", flush=True)
            print(f"[Controller] Рушій: {self.ocr_engine}, Мова: {self.language}", flush=True)
            print(f"[Controller] Шлях до зображення: {self.image_path}", flush=True)
            logger.info("[Controller] Початок розпізнавання з рушієм: %s", self.ocr_engine)
            logger.info("[Controller] Мова: %s", self.language)
            logger.info("[Controller] Шлях до зображення: %s", self.image_path)
            self.progress.emit(20, "Завантаження зображення...")
            self.progress.emit(40, f"Розпізнавання через {self.ocr_engine}...")

            try:
                print("[Controller] Виклик ocr.recognize()...", flush=True)
                text = ocr.recognize(
                    self.image_path,
                    self.language,
                    preprocess=True,
                    split_into_words=self.split_into_words
                )
                result_len = len(text) if text else 0
                print(f"[Controller] ocr.recognize() завершено, "
                      f"результат: {result_len} символів", flush=True)
                self.progress.emit(80, "Обробка результату...")
                logger.info("[Controller] Результат від рушія '%s': %s символів",
                           self.ocr_engine, len(text))
                if text:
                    logger.info("[Controller] Перші 100 символів: %s...", text[:100])
                    print(f"[Controller] Перші 100 символів: {text[:100]}...", flush=True)
                else:
                    print("[Controller] ⚠️ РЕЗУЛЬТАТ ПОРОЖНІЙ!", flush=True)
            except Exception as recognize_error:
                print(f"[Controller] ✗ Помилка в ocr.recognize(): "
                      f"{recognize_error}", flush=True)
                import traceback
                traceback_str = traceback.format_exc()
                print(f"[Controller] Traceback:\n{traceback_str}", flush=True)
                logger.error("[Controller] ✗ Помилка в ocr.recognize(): %s",
                           recognize_error, exc_info=True)
                raise

            if not text or text.strip() == "":
                logger.warning("[Controller] ⚠️ РЕЗУЛЬТАТ ПОРОЖНІЙ від рушія '%s'", self.ocr_engine)
                # Спробуємо fallback на інший рушій
                if config.is_fallback_enabled():
                    logger.warning("[Controller] ⚠️ УВІМКНЕНО FALLBACK - спроба інших рушіїв")
                    available_engines = ocr.get_available_engines()
                    logger.info("[Controller] Доступні рушії для fallback: %s", available_engines)
                    for fallback_engine in available_engines:
                        if fallback_engine != self.ocr_engine:
                            try:
                                logger.warning("[Controller] ⚠️ FALLBACK: Спроба рушія %s", fallback_engine)
                                ocr.switch_engine(fallback_engine)
                                fallback_text = ocr.recognize(
                                    self.image_path,
                                    self.language,
                                    preprocess=True,
                                    split_into_words=self.split_into_words
                                )
                                logger.warning(
                                    "[Controller] ⚠️ FALLBACK: Результат від %s: %s символів",
                                    fallback_engine, len(fallback_text))
                                if fallback_text and fallback_text.strip():
                                    text = fallback_text
                                    logger.warning(
                                        "[Controller] ⚠️ FALLBACK: Використовується "
                                        "результат від %s замість %s",
                                        fallback_engine, self.ocr_engine)
                                    break
                            except Exception as fallback_error:
                                logger.error("[Controller] ✗ Fallback на %s не вдався: %s",
                                           fallback_engine, fallback_error)
                                continue

                if not text or text.strip() == "":
                    self.error.emit("Не вдалося розпізнати текст. Спробуйте інше зображення або змініть налаштування.")
                else:
                    self.finished.emit(text)
            else:
                self.finished.emit(text)

        except Exception as e:
            import traceback
            error_msg = f"Помилка при розпізнаванні: {e}"
            traceback_str = traceback.format_exc()
            print(f"[Controller] ✗ {error_msg}", flush=True)
            print(f"[Controller] Traceback:\n{traceback_str}", flush=True)
            logger.error("[Controller] ✗ %s", error_msg, exc_info=True)
            # Записуємо в файл для діагностики
            try:
                with open("controller_error.log", "a", encoding="utf-8") as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Помилка при розпізнаванні: {e}\n")
                    f.write(f"Рушій: {self.ocr_engine}, Мова: {self.language}\n")
                    f.write(f"Traceback:\n{traceback_str}\n")
                    f.write(f"{'='*80}\n")
            except Exception:
                pass
            self.error.emit(f"Помилка при розпізнаванні: {str(e)}")


class Controller:
    """Контролер програми"""

    def __init__(self, main_window):
        self.main_window = main_window
        self.recognition_thread = None

    def recognize_text(self, image_path, language, ocr_engine, split_into_words=False):
        """Запуск розпізнавання тексту"""
        # Мапінг мов
        lang_map = {
            "Українська": "ukr",
            "Англійська": "eng"
        }

        # Мапінг рушіїв
        engine_map = {
            "Tesseract": "tesseract",
            "EasyOCR": "easyocr",
            "PaddleOCR": "paddleocr"
        }

        lang_code = lang_map.get(language, "eng")
        engine_name = engine_map.get(ocr_engine, "tesseract")

        # Створення потоку
        self.recognition_thread = RecognitionThread(
            image_path,
            lang_code,
            engine_name,
            split_into_words=split_into_words
        )
        self.recognition_thread.finished.connect(self.main_window.on_recognition_complete)
        self.recognition_thread.error.connect(self.main_window.on_recognition_error)
        if hasattr(self.main_window, 'on_progress_updated'):
            self.recognition_thread.progress.connect(self.main_window.on_progress_updated)
        self.recognition_thread.start()

    def recognize_text_async(self, image_path, engine, language):
        """Асинхронне розпізнавання з новим API"""
        from model.ocr_manager import OCREngine, OCRLanguage

        # Конвертація enum в рядки
        engine_map = {
            OCREngine.TESSERACT: "tesseract",
            OCREngine.EASYOCR: "easyocr",
            OCREngine.PADDLEOCR: "paddleocr"
        }

        lang_map = {
            OCRLanguage.UKRAINIAN: "ukr",
            OCRLanguage.ENGLISH: "eng",
            OCRLanguage.BOTH: "eng"  # Для "Обидві" використовуємо англійську
        }

        engine_name = engine_map.get(engine, "tesseract")
        lang_code = lang_map.get(language, "ukr")

        # Використовуємо існуючий метод
        lang_display = "Українська" if lang_code == "ukr" else "Англійська"
        self.recognize_text(image_path, lang_display, engine_name.capitalize())

    def get_available_engines(self):
        """Отримання списку доступних рушіїв"""
        from model.unified_ocr_adapter import UnifiedOCRAdapter
        temp_adapter = UnifiedOCRAdapter(engine='tesseract')
        return temp_adapter.get_available_engines()

    def export_text(self, text, file_path):
        """Експорт тексту у файл"""
        try:
            exporter = TextExporter()

            if file_path.endswith('.txt'):
                return exporter.export_txt(text, file_path)
            if file_path.endswith('.docx'):
                return exporter.export_docx(text, file_path)
            if file_path.endswith('.pdf'):
                return exporter.export_pdf(text, file_path)
            # За замовчуванням TXT
            return exporter.export_txt(text, file_path + '.txt')

        except Exception as e:
            print(f"Помилка експорту: {e}")
            return False
