"""
Багатопотоковий контролер з прогрес-баром
"""
from PyQt6.QtCore import QThread, pyqtSignal
from typing import Optional
import cv2
import numpy as np
import logging

from model.ocr_manager import OCRManager, OCREngine, OCRLanguage
from model.improved_llm_postprocessor import ImprovedLLMPostProcessor
from model.optimized_preprocessor import OptimizedPreprocessor
from model.handwrite_export import TextExporter

logger = logging.getLogger(__name__)

# Константи для назв рушіїв OCR
TESSERACT_DISPLAY_NAME = 'Tesseract OCR'
EASYOCR_DISPLAY_NAME = 'EasyOCR'
PADDLEOCR_DISPLAY_NAME = 'PaddleOCR'


class RecognitionWorker(QThread):
    """Робочий потік для розпізнавання"""
    
    # Сигнали
    progress_updated = pyqtSignal(int, str)  # (відсоток, повідомлення)
    finished = pyqtSignal(str, str)  # (виправлений текст, назва рушія)
    error = pyqtSignal(str)  # повідомлення про помилку
    
    def __init__(self, 
                 image_path: str,
                 engine: OCREngine,
                 language: OCRLanguage,
                 use_ai_correction: bool = False,
                 use_best_engine: bool = False,
                 llm_config: Optional[dict] = None):
        super().__init__()
        self.image_path = image_path
        self.engine = engine
        self.language = language
        self.use_ai_correction = use_ai_correction
        self.use_best_engine = use_best_engine
        self.llm_config = llm_config or {}
        self.processed_image = None  # Зберігаємо оброблене зображення для ШІ
    
    def run(self):
        """Виконання розпізнавання"""
        try:
            # Крок 1: Завантаження зображення (10%)
            self.progress_updated.emit(10, "Завантаження зображення...")
            image = cv2.imread(self.image_path)
            if image is None:
                raise ValueError(f"Не вдалося завантажити: {self.image_path}")
            
            # Крок 2: Препроцесинг (20%)
            self.progress_updated.emit(20, "Попередня обробка...")
            preprocessor = OptimizedPreprocessor()
            processed_image = preprocessor.process(image)
            self.processed_image = processed_image  # Зберігаємо для ШІ
            
            ocr_manager = OCRManager()
            ocr_manager.set_language(self.language)
            
            # Крок 3: OCR розпізнавання
            if self.use_best_engine:
                # Режим "найкращий рушій" - запускаємо всі доступні рушії
                self.progress_updated.emit(30, "Розпізнавання всіма рушіями...")
                available_engines = ocr_manager.get_available_engines()
                
                logger.info(f"[RecognitionWorker] Режим 'найкращий рушій': запуск {len(available_engines)} рушіїв")
                
                results = {}  # {engine: text}
                total_engines = len(available_engines)
                
                for idx, engine in enumerate(available_engines):
                    engine_display = "Unknown"  # Дефолтне значення
                    try:
                        progress = 30 + int((idx / total_engines) * 40)  # 30-70%
                        engine_name = engine.value
                        engine_display_names = {
                            'tesseract': TESSERACT_DISPLAY_NAME,
                            'easyocr': EASYOCR_DISPLAY_NAME,
                            'paddleocr': PADDLEOCR_DISPLAY_NAME
                        }
                        engine_display = engine_display_names.get(engine_name, engine_name.title())
                        self.progress_updated.emit(progress, f"Розпізнавання ({engine_display})...")
                        
                        ocr_manager.set_engine(engine)
                        text = ocr_manager.recognize(processed_image)
                        
                        if text and text.strip():
                            results[engine] = text
                            logger.info(f"[RecognitionWorker] {engine_display}: {len(text)} символів")
                        else:
                            logger.warning(f"[RecognitionWorker] {engine_display}: порожній результат")
                    except Exception as e:
                        logger.error(f"[RecognitionWorker] Помилка {engine_display}: {e}")
                        continue
                
                if not results:
                    self.error.emit("Не вдалося розпізнати текст жодним рушієм")
                    return
                
                # Крок 4: Вибір найкращого результату через ШІ (70-90%)
                self.progress_updated.emit(70, "Вибір найкращого результату через ШІ...")
                text, best_engine = self._select_best_result(results)
                
                # Крок 5: AI корекція (опціонально) (90-95%)
                if self.use_ai_correction:
                    self.progress_updated.emit(90, "AI корекція...")
                    try:
                        llm_processor = ImprovedLLMPostProcessor(
                            api_type=self.llm_config.get('api_type', 'ollama'),
                            api_key=self.llm_config.get('api_key'),
                            api_url=self.llm_config.get('api_url'),
                            model=self.llm_config.get('model')
                        )
                        
                        if llm_processor.is_available():
                            lang_name = "ukrainian" if self.language == OCRLanguage.UKRAINIAN else "english"
                            text = llm_processor.correct_text(text, lang_name)
                        else:
                            text = ImprovedLLMPostProcessor.simple_correction(text)
                    except Exception as e:
                        logger.warning(f"AI корекція не вдалася: {e}")
                        text = ImprovedLLMPostProcessor.simple_correction(text)
                else:
                    text = ImprovedLLMPostProcessor.simple_correction(text)
                
                # Завершення (100%)
                self.progress_updated.emit(100, "Завершено!")
                engine_display_names = {
                    'tesseract': TESSERACT_DISPLAY_NAME,
                    'easyocr': 'EasyOCR',
                    'paddleocr': 'PaddleOCR'
                }
                best_engine_display = engine_display_names.get(best_engine.value, best_engine.value.title())
                self.finished.emit(text, f"{best_engine_display} (найкращий з {len(results)})")
            else:
                # Звичайний режим - один рушій
                self.progress_updated.emit(60, f"Розпізнавання ({self.engine.value})...")
                ocr_manager.set_engine(self.engine)
                text = ocr_manager.recognize(processed_image)
                
                if not text or not text.strip():
                    self.error.emit("Не вдалося розпізнати текст")
                    return
                
                # Крок 4: AI корекція (опціонально) (90%)
                if self.use_ai_correction:
                    self.progress_updated.emit(90, "AI корекція...")
                    try:
                        llm_processor = ImprovedLLMPostProcessor(
                            api_type=self.llm_config.get('api_type', 'ollama'),
                            api_key=self.llm_config.get('api_key'),
                            api_url=self.llm_config.get('api_url'),
                            model=self.llm_config.get('model')
                        )
                        
                        if llm_processor.is_available():
                            lang_name = "ukrainian" if self.language == OCRLanguage.UKRAINIAN else "english"
                            text = llm_processor.correct_text(text, lang_name)
                        else:
                            text = ImprovedLLMPostProcessor.simple_correction(text)
                    except Exception as e:
                        logger.warning(f"AI корекція не вдалася: {e}")
                        text = ImprovedLLMPostProcessor.simple_correction(text)
                else:
                    text = ImprovedLLMPostProcessor.simple_correction(text)
                
                # Завершення (100%)
                self.progress_updated.emit(100, "Завершено!")
                engine_name = self.engine.value
                engine_display_names = {
                    'tesseract': TESSERACT_DISPLAY_NAME,
                    'easyocr': 'EasyOCR',
                    'paddleocr': 'PaddleOCR'
                }
                engine_display = engine_display_names.get(engine_name, engine_name.title())
                self.finished.emit(text, engine_display)
            
        except Exception as e:
            logger.error(f"Помилка розпізнавання: {e}", exc_info=True)
            self.error.emit(f"Помилка: {str(e)}")
    
    def _select_best_result(self, results: dict, image: Optional[np.ndarray] = None) -> tuple[str, OCREngine]:
        """Вибір найкращого результату через ШІ з попередньою фільтрацією"""
        if len(results) == 1:
            engine = list(results.keys())[0]
            return results[engine], engine
        
        if len(set(results.keys())) == 1:
            engine = list(results.keys())[0]
            logger.info(f"[RecognitionWorker] Всі результати від одного рушія ({engine.value}), пропускаємо ШІ")
            print(f"[RecognitionWorker] Всі результати від одного рушія ({engine.value}), пропускаємо ШІ", flush=True)
            return results[engine], engine
        
        filtered_results = self._filter_results(results)
        
        if not filtered_results:
            logger.warning("[RecognitionWorker] Всі результати відфільтровані, використовуємо всі результати")
            filtered_results = results
        
        if len(filtered_results) == 1:
            engine = list(filtered_results.keys())[0]
            logger.info(f"[RecognitionWorker] Після фільтрації залишився один результат від {engine.value}")
            return filtered_results[engine], engine
        
        return self._evaluate_and_select_best(filtered_results, results)
    
    def _filter_results(self, results: dict) -> dict:
        """Фільтрація результатів перед відправкою в ШІ"""
        filtered_results = {}
        
        for engine, text in results.items():
            if not text or not text.strip():
                continue
            
            if not self._should_include_result(engine, text):
                continue
            
            filtered_results[engine] = text
        
        return filtered_results
    
    def _should_include_result(self, engine: OCREngine, text: str) -> bool:
        """Визначення, чи слід включити результат у фільтрацію"""
        if self.language == OCRLanguage.UKRAINIAN:
            if not self._check_ukrainian_cyrillic(engine, text):
                return False
        
        if not self._check_unreadable_chars(engine, text):
            return False
        
        if not self._check_min_length(engine, text):
            return False
        
        if not self._check_letter_ratio(engine, text):
            return False
        
        return True
    
    def _check_ukrainian_cyrillic(self, engine: OCREngine, text: str) -> bool:
        """Перевірка наявності кирилиці для української мови"""
        has_cyrillic = any('\u0400' <= char <= '\u04FF' for char in text)
        
        if not has_cyrillic:
            latin_chars = sum(1 for char in text if 'a' <= char.lower() <= 'z')
            latin_ratio = latin_chars / len(text) if text else 0
            
            if latin_ratio > 0.5:
                logger.warning(f"[RecognitionWorker] Пропущено результат від {engine.value}: '{text}' (немає кирилиці для української, {latin_ratio:.0%} латиниці)")
                return False
            
            if len(text.strip()) <= 4 and latin_ratio > 0.3:
                logger.warning(f"[RecognitionWorker] Пропущено результат від {engine.value}: '{text}' (занадто короткий без кирилиці)")
                return False
        
        return True
    
    def _check_unreadable_chars(self, engine: OCREngine, text: str) -> bool:
        """Перевірка наявності нечитабельних символів"""
        unreadable_chars = sum(1 for char in text if char in '|\\/[]{}()<>')
        if unreadable_chars > len(text.strip()) * 0.3:
            logger.warning(f"[RecognitionWorker] Пропущено результат від {engine.value}: '{text}' (багато нечитабельних символів)")
            return False
        return True
    
    def _check_min_length(self, engine: OCREngine, text: str) -> bool:
        """Перевірка мінімальної довжини тексту"""
        min_length = 3 if self.language == OCRLanguage.ENGLISH else 2
        if len(text.strip()) < min_length:
            logger.warning(f"[RecognitionWorker] Пропущено результат від {engine.value}: '{text}' (занадто короткий, мінімум {min_length} символи)")
            return False
        return True
    
    def _check_letter_ratio(self, engine: OCREngine, text: str) -> bool:
        """Перевірка відсотка букв у тексті"""
        letter_count = sum(1 for char in text if char.isalpha())
        if len(text.strip()) > 0:
            letter_ratio = letter_count / len(text.strip())
            if letter_ratio < 0.3:
                logger.warning(f"[RecognitionWorker] Пропущено результат від {engine.value}: '{text}' (замало букв: {letter_ratio:.1%})")
                return False
        return True
    
    def _evaluate_and_select_best(self, filtered_results: dict, all_results: dict) -> tuple[str, OCREngine]:
        """Оцінка та вибір найкращого результату"""
        logger.info("[RecognitionWorker] Використання fallback вибору за очками...")
        print("[RecognitionWorker] Використання fallback вибору за очками...", flush=True)
        
        results_to_score = filtered_results if filtered_results else all_results
        all_lengths = [len(t.strip()) for t in results_to_score.values()] if results_to_score else []
        max_len = max(all_lengths) if all_lengths else 0
        
        scored_results = []
        for engine, text in results_to_score.items():
            score = self._score_result(text, filtered_results, all_results)
            
            if self.language == OCRLanguage.UKRAINIAN and len(results_to_score) > 1:
                text_len = len(text.strip())
                if text_len < 5 and max_len > text_len + 1:
                    score -= 30
                    logger.debug(f"[RecognitionWorker] Штраф за короткий результат '{text}' (довжина: {text_len}, макс: {max_len})")
            
            scored_results.append((engine, text, score))
        
        scored_results.sort(key=lambda x: x[2], reverse=True)
        
        if not scored_results:
            if all_results:
                engine = list(all_results.keys())[0]
                logger.warning(f"[RecognitionWorker] Всі результати відфільтровані, використовується перший: {engine.value}")
                return all_results[engine], engine
            return "", OCREngine.TESSERACT
        
        best_engine, best_text, best_score = scored_results[0]
        logger.info(f"[RecognitionWorker] Fallback: вибрано результат від {best_engine.value} з оцінкою {best_score:.1f}")
        logger.info(f"[RecognitionWorker] Fallback: вибраний текст: '{best_text}'")
        logger.info(f"[RecognitionWorker] Всі оцінки: {[(e.value, t, f'{s:.1f}') for e, t, s in scored_results]}")
        print(f"[RecognitionWorker] Fallback: вибрано '{best_text}' від {best_engine.value} (оцінка: {best_score:.1f})", flush=True)
        print(f"[RecognitionWorker] Всі результати: {[(e.value, t[:20], f'{s:.1f}') for e, t, s in scored_results]}", flush=True)
        
        return best_text, best_engine
    
    def _score_result(self, text: str, filtered_results: dict, all_results: dict) -> float:
        """Оцінка якості результату"""
        score = 0.0
        text_stripped = text.strip()
        
        score += len(text_stripped) * 0.1
        
        if self.language == OCRLanguage.UKRAINIAN:
            score += self._score_ukrainian_text(text_stripped, filtered_results, all_results)
        
        score += self._score_readability(text_stripped, filtered_results, all_results)
        score += self._score_text_quality(text_stripped)
        
        return score
    
    def _score_ukrainian_text(self, text_stripped: str, filtered_results: dict, all_results: dict) -> float:
        """Оцінка українського тексту"""
        score = 0.0
        has_cyrillic = any('\u0400' <= char <= '\u04FF' for char in text_stripped)
        has_latin = any(char.isalpha() and ord(char) < 128 and not ('\u0400' <= char <= '\u04FF') for char in text_stripped)
        
        if has_cyrillic and has_latin:
            score -= 150
            logger.debug(f"[RecognitionWorker] Штраф за змішані символи: '{text_stripped}'")
        
        if has_cyrillic:
            score += 200
            score += self._score_ukrainian_special_chars(text_stripped)
            score += self._score_ukrainian_common_words(text_stripped)
            score += self._score_ukrainian_word_patterns(text_stripped)
        else:
            score -= 100
        
        return score
    
    def _score_ukrainian_special_chars(self, text_stripped: str) -> float:
        """Оцінка за українські особливі символи"""
        uk_special_chars = sum(1 for char in text_stripped if char in 'іїєІЇЄ')
        score = uk_special_chars * 50
        
        if 'ї' in text_stripped or 'Ї' in text_stripped:
            score += 25
        
        return score
    
    def _score_ukrainian_common_words(self, text_stripped: str) -> float:
        """Оцінка за поширені українські слова"""
        score = 0.0
        common_uk_words = ['привіт', 'проба', 'тест', 'добрий', 'день', 'вітаю', 'дякую', 'спроба', 'любов', 'україна']
        text_lower = text_stripped.lower()
        text_no_spaces = text_lower.replace(' ', '').replace('\n', '')
        
        for word in common_uk_words:
            if word in text_lower:
                score += 150
                if text_lower == word or text_lower.startswith(word):
                    score += 50
                
                if word == 'україна':
                    score += self._score_ukraine_word(text_no_spaces)
                elif word == 'привіт':
                    score += self._score_privit_word(text_no_spaces)
                break
        
        return score
    
    def _score_ukraine_word(self, text_no_spaces: str) -> float:
        """Оцінка за слово 'україна'"""
        score = 0.0
        
        if 'укр' in text_no_spaces and ('країна' in text_no_spaces or 'країно' in text_no_spaces):
            score += 100
        
        if 'інб' in text_no_spaces:
            score -= 50
        
        text_no_special = text_no_spaces.replace('|', '').replace('\\', '').replace('/', '').replace('(', '').replace(')', '')
        if 'країна' in text_no_special.lower() or 'країно' in text_no_special.lower():
            score += 60
            if text_no_special.lower().startswith('укр') or 'укр' in text_no_special.lower():
                score += 40
            if 'інб' in text_no_special.lower():
                score -= 50
            if 'їна' in text_no_special.lower() or 'їно' in text_no_special.lower():
                score += 30
        
        return score
    
    def _score_privit_word(self, text_no_spaces: str) -> float:
        """Оцінка за слово 'привіт'"""
        score = 0.0
        
        if 'при' in text_no_spaces and 'віт' in text_no_spaces:
            score += 100
        
        if 'гб' in text_no_spaces or 'гбіт' in text_no_spaces:
            score -= 80
        
        return score
    
    def _score_ukrainian_word_patterns(self, text_stripped: str) -> float:
        """Оцінка за патерни українських слів"""
        score = 0.0
        text_lower = text_stripped.lower()
        text_no_spaces = text_lower.replace(' ', '').replace('\n', '')
        text_no_special = text_no_spaces.replace('|', '').replace('\\', '').replace('/', '').replace('(', '').replace(')', '')
        
        if text_stripped.startswith('Пр') or text_stripped.startswith('пр'):
            score += 50
            if 'віт' in text_lower or 'віт' in text_no_spaces:
                score += 40
            if 'гб' in text_no_spaces:
                score -= 60
        
        if 'іт' in text_lower or 'Іт' in text_stripped:
            score += 30
        
        if 'рифт' in text_no_special.lower() or 'віт' in text_no_special.lower():
            score += 50
            if text_no_special.lower().startswith('при') or 'при' in text_no_special.lower():
                score += 30
        
        if 'юбов' in text_no_special.lower() or 'юбов' in text_lower:
            score += 60
            if text_no_special.lower().startswith('лю') or 'лю' in text_no_special.lower():
                score += 40
            if text_stripped.startswith('І') or text_stripped.startswith('і'):
                score += 20
        
        return score
    
    def _score_readability(self, text_stripped: str, filtered_results: dict, all_results: dict) -> float:
        """Оцінка читабельності тексту"""
        score = 0.0
        has_cyrillic = any('\u0400' <= char <= '\u04FF' for char in text_stripped)
        unreadable_chars = sum(1 for char in text_stripped if char in '|\\/[]{}()<>')
        
        if self.language == OCRLanguage.UKRAINIAN and has_cyrillic:
            text_no_special = text_stripped.replace('|', '').replace('\\', '').replace('/', '').replace('(', '').replace(')', '')
            if 'рифт' in text_no_special.lower() or 'віт' in text_no_special.lower():
                score -= unreadable_chars * 10
            else:
                score -= unreadable_chars * 20
        else:
            score -= unreadable_chars * 20
        
        if '\n' in text_stripped:
            if len(text_stripped) < 10:
                score -= 40
            else:
                score -= 20
        
        punctuation_ratio = sum(1 for char in text_stripped if char in '.,;:!?') / max(len(text_stripped), 1)
        if punctuation_ratio > 0.3:
            score -= 50
        
        if text_stripped.endswith('.') and len(text_stripped) < 15:
            score -= 30
            if self.language == OCRLanguage.UKRAINIAN:
                other_results = [t for e, t in (filtered_results if filtered_results else all_results).items() if t != text_stripped]
                if other_results:
                    results_without_dot = [t for t in other_results if not t.strip().endswith('.')]
                    if results_without_dot:
                        score -= 20
                        logger.debug(f"[RecognitionWorker] Штраф за крапку в кінці '{text_stripped}' (є альтернативи без крапки)")
        
        return score
    
    def _score_text_quality(self, text_stripped: str) -> float:
        """Оцінка загальної якості тексту"""
        score = 0.0
        
        letter_count = sum(1 for char in text_stripped if char.isalpha())
        if letter_count > 0:
            letter_ratio = letter_count / len(text_stripped)
            score += letter_ratio * 100
        
        if len(text_stripped) < 3:
            score -= 50
        
        if len(text_stripped) > 0:
            special_char_ratio = sum(1 for char in text_stripped if not (char.isalnum() or char.isspace())) / len(text_stripped)
            if special_char_ratio > 0.4:
                score -= 100
        
        return score


class ImprovedController:
    """Покращений контролер з багатопотоковістю"""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.worker = None
        self.ocr_manager = OCRManager()
    
    def get_available_engines(self) -> list[OCREngine]:
        """Отримання доступних рушіїв"""
        return self.ocr_manager.get_available_engines()
    
    def get_engine_info(self, engine: OCREngine) -> dict:
        """Отримання інформації про рушій"""
        metadata = self.ocr_manager.get_engine_metadata(engine)
        if metadata:
            return {
                'name': metadata.name,
                'description': metadata.description,
                'speed': metadata.speed,
                'accuracy': metadata.accuracy,
                'best_for': metadata.best_for
            }
        return {}
    
    def recognize_text_async(self,
                            image_path: str,
                            engine: OCREngine,
                            language: OCRLanguage,
                            use_ai_correction: bool = False,
                            use_best_engine: bool = False,
                            llm_config: Optional[dict] = None):
        """Асинхронне розпізнавання тексту"""
        
        # Перевіряємо, чи не виконується вже розпізнавання
        if self.worker and self.worker.isRunning():
            logger.warning("Розпізнавання вже виконується")
            return
        
        # Створюємо та запускаємо робочий потік
        self.worker = RecognitionWorker(
            image_path=image_path,
            engine=engine,
            language=language,
            use_ai_correction=use_ai_correction,
            use_best_engine=use_best_engine,
            llm_config=llm_config
        )
        
        # Підключаємо сигнали
        self.worker.progress_updated.connect(self.main_window.on_progress_updated)
        self.worker.finished.connect(self.main_window.on_recognition_complete)
        self.worker.error.connect(self.main_window.on_recognition_error)
        
        # Запускаємо
        self.worker.start()
    
    def export_text(self, text: str, file_path: str) -> bool:
        """Експорт тексту у файл"""
        try:
            exporter = TextExporter()
            
            if file_path.endswith('.txt'):
                return exporter.export_txt(text, file_path)
            elif file_path.endswith('.docx'):
                return exporter.export_docx(text, file_path)
            elif file_path.endswith('.pdf'):
                return exporter.export_pdf(text, file_path)
            else:
                return exporter.export_txt(text, file_path + '.txt')
                
        except Exception as e:
            logger.error(f"Помилка експорту: {e}")
            return False

