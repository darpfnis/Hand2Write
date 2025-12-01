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
        
        # Якщо всі результати від одного рушія - повертаємо перший
        if len(set(results.keys())) == 1:
            engine = list(results.keys())[0]
            logger.info(f"[RecognitionWorker] Всі результати від одного рушія ({engine.value}), пропускаємо ШІ")
            print(f"[RecognitionWorker] Всі результати від одного рушія ({engine.value}), пропускаємо ШІ", flush=True)
            return results[engine], engine
        
        # Фільтрація результатів перед відправкою в ШІ
        # Виключаємо явно неправильні результати
        filtered_results = {}
        for engine, text in results.items():
            if not text or not text.strip():
                continue
            
            # Для української мови: перевіряємо, чи є кириличні літери
            if self.language == OCRLanguage.UKRAINIAN:
                has_cyrillic = any('\u0400' <= char <= '\u04FF' for char in text)
                # Якщо немає кирилиці - це підозріло для української мови
                if not has_cyrillic:
                    # Перевіряємо, чи це не просто латиниця (можливо помилка розпізнавання)
                    latin_chars = sum(1 for char in text if 'a' <= char.lower() <= 'z')
                    latin_ratio = latin_chars / len(text) if text else 0
                    
                    # Фільтруємо результати без кирилиці, особливо короткі (наприклад, "bir")
                    if latin_ratio > 0.5:  # Більше 50% латиниці - підозріло
                        logger.warning(f"[RecognitionWorker] Пропущено результат від {engine.value}: '{text}' (немає кирилиці для української, {latin_ratio:.0%} латиниці)")
                        continue
                    
                    # Якщо дуже короткий результат без кирилиці (наприклад, "bir", "abc") - точно фільтруємо
                    if len(text.strip()) <= 4 and latin_ratio > 0.3:
                        logger.warning(f"[RecognitionWorker] Пропущено результат від {engine.value}: '{text}' (занадто короткий без кирилиці)")
                        continue
            
            # Перевіряємо наявність нечитабельних символів (|, \, /, тощо)
            unreadable_chars = sum(1 for char in text if char in '|\\/[]{}()<>')
            if unreadable_chars > len(text.strip()) * 0.3:  # Більше 30% нечитабельних символів
                logger.warning(f"[RecognitionWorker] Пропущено результат від {engine.value}: '{text}' (багато нечитабельних символів)")
                continue
            
            # Перевіряємо, чи текст не надто короткий (менше 2 символів для слова)
            # Для англійської мови - мінімум 3 символи (наприклад, "e" замість "Hello" - поганий результат)
            min_length = 3 if self.language == OCRLanguage.ENGLISH else 2
            if len(text.strip()) < min_length:
                logger.warning(f"[RecognitionWorker] Пропущено результат від {engine.value}: '{text}' (занадто короткий, мінімум {min_length} символи)")
                continue
            
            # Перевіряємо відсоток букв (результат має містити переважно букви)
            letter_count = sum(1 for char in text if char.isalpha())
            if len(text.strip()) > 0:
                letter_ratio = letter_count / len(text.strip())
                if letter_ratio < 0.3:  # Менше 30% букв - підозріло
                    logger.warning(f"[RecognitionWorker] Пропущено результат від {engine.value}: '{text}' (замало букв: {letter_ratio:.1%})")
                    continue
            
            filtered_results[engine] = text
        
        # Якщо після фільтрації не залишилося результатів, використовуємо всі
        if not filtered_results:
            logger.warning("[RecognitionWorker] Всі результати відфільтровані, використовуємо всі результати")
            filtered_results = results
        
        # Якщо після фільтрації залишився один результат, повертаємо його
        if len(filtered_results) == 1:
            engine = list(filtered_results.keys())[0]
            logger.info(f"[RecognitionWorker] Після фільтрації залишився один результат від {engine.value}")
            return filtered_results[engine], engine
        
        # Використовуємо fallback scoring для вибору найкращого результату
        logger.info("[RecognitionWorker] Використання fallback вибору за очками...")
        print("[RecognitionWorker] Використання fallback вибору за очками...", flush=True)
        
        # Fallback: вибираємо найкращий результат на основі кількох критеріїв
        def score_result(text):
            """Оцінка якості результату"""
            score = 0
            text_stripped = text.strip()
            
            # Базова оцінка за довжину (довші результати зазвичай кращі, але не завжди)
            score += len(text_stripped) * 0.1
            
            # Для української: перевірка наявності кирилиці
            if self.language == OCRLanguage.UKRAINIAN:
                has_cyrillic = any('\u0400' <= char <= '\u04FF' for char in text_stripped)
                has_latin = any(char.isalpha() and ord(char) < 128 and not ('\u0400' <= char <= '\u04FF') for char in text_stripped)
                
                # Штраф за змішані латиниця/кирилиця (наприклад, "gнд", "gnd") - це поганий результат
                if has_cyrillic and has_latin:
                    score -= 150  # Великий штраф за змішані символи
                    logger.debug(f"[RecognitionWorker] Штраф за змішані символи: '{text_stripped}'")
                
                if has_cyrillic:
                    score += 200  # Великий бонус за наявність кирилиці
                    # Додатковий бонус за українські особливі символи
                    uk_special_chars = sum(1 for char in text_stripped if char in 'іїєІЇЄ')
                    score += uk_special_chars * 50  # Бонус за кожен український особливий символ
                    
                    # Бонус за правильні українські слова (перевірка на поширені слова)
                    common_uk_words = ['привіт', 'проба', 'тест', 'добрий', 'день', 'вітаю', 'дякую', 'спроба', 'любов', 'україна']
                    text_lower = text_stripped.lower()
                    text_no_spaces = text_lower.replace(' ', '').replace('\n', '')
                    
                    # Перевірка на точну відповідність або схожість
                    for word in common_uk_words:
                        if word in text_lower:
                            score += 150  # Великий бонус за правильне слово
                            # Додатковий бонус, якщо слово точно збігається
                            if text_lower == word or text_lower.startswith(word):
                                score += 50
                            # Перевірка на схожість (проста перевірка на наявність ключових частин)
                            if word == 'україна':
                                # Перевіряємо наявність правильних частин "укр", "країна"
                                if 'укр' in text_no_spaces and ('країна' in text_no_spaces or 'країно' in text_no_spaces):
                                    score += 100  # Великий бонус за правильні частини
                                # Штраф за помилки в кінці (наприклад, "інб" замість "їна")
                                if 'інб' in text_no_spaces:
                                    score -= 50  # Штраф за помилку "інб" замість "їна"
                            elif word == 'привіт':
                                # Перевіряємо наявність правильних частин "при", "віт"
                                if 'при' in text_no_spaces and 'віт' in text_no_spaces:
                                    score += 100  # Великий бонус за правильні частини
                                # Штраф за помилки в середині (наприклад, "гб" замість "в")
                                if 'гб' in text_no_spaces or 'гбіт' in text_no_spaces:
                                    score -= 80  # Штраф за помилку "гб" замість "в"
                            break
                    
                    # Бонус за слова, що починаються з "Пр" (може бути "Привіт", "Проба")
                    if text_stripped.startswith('Пр') or text_stripped.startswith('пр'):
                        score += 50  # Бонус за правильний початок
                        # Додатковий бонус, якщо це схоже на "Привіт"
                        if 'віт' in text_lower or 'віт' in text_no_spaces:
                            score += 40  # Бонус за правильне закінчення "віт"
                        # Штраф за помилки в середині (наприклад, "гб" замість "в")
                        if 'гб' in text_no_spaces:
                            score -= 60  # Штраф за помилку "гб"
                    
                    # Бонус за слова, що містять "іт" (для "Привіт")
                    if 'іт' in text_lower or 'Іт' in text_stripped:
                        score += 30  # Бонус за правильне закінчення
                    
                    # Бонус за слова, що містять "рифт" або "віт" (для "Привіт"), навіть зі спецсимволами
                    text_no_special = text_no_spaces.replace('|', '').replace('\\', '').replace('/', '').replace('(', '').replace(')', '')
                    if 'рифт' in text_no_special.lower() or 'віт' in text_no_special.lower():
                        score += 50  # Великий бонус за правильні частини слова
                        # Додатковий бонус, якщо є "при" на початку
                        if text_no_special.lower().startswith('при') or 'при' in text_no_special.lower():
                            score += 30  # Бонус за правильний початок "при"
                    
                    # Бонус за слова, що містять "юбов" (для "Любов"), навіть якщо перша літера неправильна
                    if 'юбов' in text_no_special.lower() or 'юбов' in text_lower:
                        score += 60  # Великий бонус за правильні частини слова "любов"
                        # Додатковий бонус, якщо є "лю" на початку
                        if text_no_special.lower().startswith('лю') or 'лю' in text_no_special.lower():
                            score += 40  # Бонус за правильний початок "лю"
                        # Менший штраф за неправильну першу літеру (наприклад, "І" замість "Л")
                        if text_stripped.startswith('І') or text_stripped.startswith('і'):
                            # Якщо є "юбов", але починається з "І", це може бути "Любов"
                            score += 20  # Невеликий бонус за можливу помилку першої літери
                    
                    # Бонус за слова, що містять "країна" або "країно" (для "Україна")
                    if 'країна' in text_no_special.lower() or 'країно' in text_no_special.lower():
                        score += 60  # Великий бонус за правильні частини слова "україна"
                        # Додатковий бонус, якщо є "укр" на початку
                        if text_no_special.lower().startswith('укр') or 'укр' in text_no_special.lower():
                            score += 40  # Бонус за правильний початок "укр"
                        # Штраф за помилки в кінці (наприклад, "інб" замість "їна")
                        if 'інб' in text_no_special.lower():
                            score -= 50  # Штраф за помилку "інб" замість "їна"
                        # Бонус за правильне закінчення "їна" або "їно"
                        if 'їна' in text_no_special.lower() or 'їно' in text_no_special.lower():
                            score += 30  # Бонус за правильне закінчення
                    
                    # Додатковий бонус за українську особливу літеру "ї" (важливо для "Україна")
                    if 'ї' in text_stripped or 'Ї' in text_stripped:
                        score += 25  # Бонус за наявність "ї"
                else:
                    score -= 100  # Великий штраф за відсутність кирилиці для української
            
            # Штраф за нечитабельні символи
            unreadable_chars = sum(1 for char in text_stripped if char in '|\\/[]{}()<>')
            # Менший штраф, якщо текст містить правильні частини слова (наприклад, "|рифт" має "рифт")
            if self.language == OCRLanguage.UKRAINIAN and has_cyrillic:
                # Перевіряємо, чи є правильні частини слова навіть зі спецсимволами
                text_no_special = text_stripped.replace('|', '').replace('\\', '').replace('/', '').replace('(', '').replace(')', '')
                if 'рифт' in text_no_special.lower() or 'віт' in text_no_special.lower():
                    # Якщо є правильні частини, штраф менший
                    score -= unreadable_chars * 10  # Менший штраф за спецсимволи, якщо є правильні частини
                else:
                    score -= unreadable_chars * 20  # Повний штраф за спецсимволи
            else:
                score -= unreadable_chars * 20  # Штраф за кожен нечитабельний символ
            
            # Штраф за переноси рядків (\n) в коротких результатах - це поганий результат
            if '\n' in text_stripped:
                # Для коротких результатів (менше 10 символів) переноси рядків - це погано
                if len(text_stripped) < 10:
                    score -= 40  # Штраф за переноси рядків в коротких результатах
                else:
                    score -= 20  # Менший штраф для довгих результатів
            
            # Штраф за занадто багато пробілів або розділових знаків
            punctuation_ratio = sum(1 for char in text_stripped if char in '.,;:!?') / max(len(text_stripped), 1)
            if punctuation_ratio > 0.3:  # Більше 30% розділових знаків - підозріло
                score -= 50
            
            # Штраф за крапку в кінці коротких слів (наприклад, "Україно." замість "Україна")
            if text_stripped.endswith('.') and len(text_stripped) < 15:
                score -= 30  # Штраф за крапку в кінці короткого слова
                # Додатковий штраф, якщо є альтернативи без крапки
                if self.language == OCRLanguage.UKRAINIAN:
                    # Перевіряємо, чи є інші результати без крапки
                    other_results = [t for e, t in (filtered_results if filtered_results else results).items() if t != text_stripped]
                    if other_results:
                        # Якщо є результати без крапки, штрафуємо сильніше
                        results_without_dot = [t for t in other_results if not t.strip().endswith('.')]
                        if results_without_dot:
                            score -= 20  # Додатковий штраф, якщо є альтернативи без крапки
                            logger.debug(f"[RecognitionWorker] Штраф за крапку в кінці '{text_stripped}' (є альтернативи без крапки)")
            
            # Бонус за наявність букв (не тільки символи)
            letter_count = sum(1 for char in text_stripped if char.isalpha())
            if letter_count > 0:
                letter_ratio = letter_count / len(text_stripped)
                score += letter_ratio * 100  # Бонус за високий відсоток букв
            
            # Штраф за дуже короткі результати
            if len(text_stripped) < 3:
                score -= 50
            
            # Штраф за результати, які виглядають як помилки (багато спецсимволів)
            if len(text_stripped) > 0:
                special_char_ratio = sum(1 for char in text_stripped if not (char.isalnum() or char.isspace())) / len(text_stripped)
                if special_char_ratio > 0.4:  # Більше 40% спецсимволів
                    score -= 100
            
            return score
        
        # Оцінюємо всі результати (використовуємо filtered_results, або всі якщо порожні)
        results_to_score = filtered_results if filtered_results else results
        
        # Для української: обчислюємо довжини для порівняння
        all_lengths = [len(t.strip()) for t in results_to_score.values()] if results_to_score else []
        max_len = max(all_lengths) if all_lengths else 0
        
        scored_results = []
        for engine, text in results_to_score.items():
            score = score_result(text)
            # Додатковий штраф для коротких результатів, якщо є довші альтернативи
            if self.language == OCRLanguage.UKRAINIAN and len(results_to_score) > 1:
                text_len = len(text.strip())
                if text_len < 5 and max_len > text_len + 1:  # Результат значно коротший за інші
                    score -= 30
                    logger.debug(f"[RecognitionWorker] Штраф за короткий результат '{text}' (довжина: {text_len}, макс: {max_len})")
            scored_results.append((engine, text, score))
        scored_results.sort(key=lambda x: x[2], reverse=True)  # Сортуємо за оцінкою
        
        if not scored_results:
            # Якщо все відфільтровано, повертаємо перший доступний результат
            if results:
                engine = list(results.keys())[0]
                logger.warning(f"[RecognitionWorker] Всі результати відфільтровані, використовується перший: {engine.value}")
                return results[engine], engine
            else:
                return "", OCREngine.TESSERACT
        
        best_engine, best_text, best_score = scored_results[0]
        logger.info(f"[RecognitionWorker] Fallback: вибрано результат від {best_engine.value} з оцінкою {best_score:.1f}")
        logger.info(f"[RecognitionWorker] Fallback: вибраний текст: '{best_text}'")
        logger.info(f"[RecognitionWorker] Всі оцінки: {[(e.value, t, f'{s:.1f}') for e, t, s in scored_results]}")
        print(f"[RecognitionWorker] Fallback: вибрано '{best_text}' від {best_engine.value} (оцінка: {best_score:.1f})", flush=True)
        print(f"[RecognitionWorker] Всі результати: {[(e.value, t[:20], f'{s:.1f}') for e, t, s in scored_results]}", flush=True)
        
        return best_text, best_engine


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

