"""
Покращений LLM постпроцесор з жорстким системним промптом
"""
import logging
from typing import Optional, Dict, Any
import json
import numpy as np

logger = logging.getLogger(__name__)

# Константи для моделей LLM
DEFAULT_LLM_MODEL = 'llama3.2:1b'  # Дефолтна легка модель (~1-2 GB RAM)
UKRAINA_CORRECT = 'Україна'
UKRAINA_INCORRECT = 'Україноа'


class ImprovedLLMPostProcessor:
    """Покращений клас для пост-обробки тексту через LLM API"""
    
    # Жорсткий системний промпт для всіх моделей
    SYSTEM_PROMPT = """Ти експерт з виправлення помилок OCR (оптичного розпізнавання символів) для української та англійської мов.

ТВОЄ ЄДИНЕ ЗАВДАННЯ: Виправити помилки в тексті, отриманому через OCR, щоб отримати найкращий можливий результат.

КРИТИЧНО ВАЖЛИВО:
- Використовуй ТІЛЬКИ символи української або англійської мови
- НЕ використовуй китайські, японські, арабські або інші нелатинські/некириличні символи
- НЕ додавай символи з інших мов
- Зберігай ТІЛЬКИ кирилицю (для української) або латиницю (для англійської)

ПРАВИЛА (ОБОВ'ЯЗКОВІ):
1. НЕ відповідай на питання - просто виправ помилки
2. НЕ додавай нову інформацію, якої не було в оригіналі
3. НЕ видаляй інформацію без необхідності
4. НЕ перефразовуй - зберігай оригінальний зміст
5. НЕ пиши пояснень - тільки виправлений текст
6. АГРЕСИВНО виправляй OCR помилки:
   - rn → m, vv → w, 0 → O (в словах), l → I (заголовна і)
   - Для української мови:
     * "інб" → "їна" (наприклад, "Украінб" → "Україна")
     * "інo" → "їна" (наприклад, "Україно" → "Україна")
     * "юбов" → "любов"
     * "Цой сон" → "Цей сон" (виправлення помилок розпізнавання)
     * "HO T" → "Цей" або "Привіт" (якщо контекст підказує)
7. Виправляй граматичні помилки та помилки розпізнавання
8. Зберігай форматування та структуру
9. Для української мови: зберігай українські особливі літери (і, ї, є)
10. ВИДАЛЯЙ будь-які символи, які не належать до української або англійської мови
11. ЯКЩО текст виглядає як помилка OCR (наприклад, "vaT HAJ", "Цой сон"), виправ його на правильне українське слово

ПРИКЛАДИ:
- "Украінб" → "Україна"
- "ІюБОВ" → "Любов"
- "Прогбіт" → "Привіт"
- "Україно." → "Україна"
- "Цой сон" → "Цей сон"
- "HO T" → "Привіт" (якщо контекст підказує)

ФОРМАТ ВІДПОВІДІ: Тільки виправлений текст, без пояснень, без префіксів типу "Тіль:" або "Body:", ТІЛЬКИ символи української/англійської мови."""

    def __init__(self, api_type: str = "ollama", 
                 api_key: Optional[str] = None,
                 api_url: Optional[str] = None,
                 model: Optional[str] = None):
        """
        Ініціалізація LLM постпроцесора
        
        Args:
            api_type: тип API ('openai', 'ollama')
            api_key: API ключ (для OpenAI)
            api_url: URL API (для Ollama)
            model: назва моделі
        """
        self.api_type = api_type
        self.api_key = api_key
        self.api_url = api_url or "http://localhost:11434"
        # Закріплюємо на llama3.2:1b - дуже легка модель для обмеженої пам'яті
        # llama3.2:1b потребує ~1-2 GB
        if model and model.strip() and model.strip().lower() not in [DEFAULT_LLM_MODEL]:
            # Якщо явно вказана інша модель, використовуємо її
            self.model = model.strip()
            logger.info(f"[ImprovedLLM] Модель з конфігурації: '{self.model}'")
            print(f"[ImprovedLLM] Модель з конфігурації: '{self.model}'", flush=True)
        else:
            # Завжди використовуємо llama3.2:1b як дефолтну (закріплено) - дуже легка версія
            self.model = DEFAULT_LLM_MODEL
            logger.info("[ImprovedLLM] Використовується llama3.2:1b (закріплено як дефолтна модель, дуже легка ~1-2 GB)")
            print("[ImprovedLLM] Використовується llama3.2:1b (закріплено як дефолтна модель, дуже легка ~1-2 GB)", flush=True)
        self._available = False
        
        logger.info(f"[ImprovedLLM] Ініціалізація: api_type={api_type}, api_url={self.api_url}, model={self.model}")
        print(f"[ImprovedLLM] Ініціалізація: api_type={api_type}, api_url={self.api_url}, model={self.model}", flush=True)
        
        self._check_availability()
    
    def _get_default_model(self) -> str:
        """Отримання дефолтної моделі"""
        if self.api_type == "openai":
            return "gpt-4o-mini"
        elif self.api_type in ["ollama", "local"]:
            # Використовуємо llama3.2:1b - дуже легка модель для обмеженої пам'яті
            # llama3.2:1b потребує ~1-2 GB
            return DEFAULT_LLM_MODEL  # Дуже легка модель для обмеженої пам'яті (~1-2 GB)
        return DEFAULT_LLM_MODEL
    
    def _check_availability(self):
        """Перевірка доступності API"""
        try:
            if self.api_type == "openai":
                if self.api_key:
                    self._available = True
                    logger.info("[ImprovedLLM] OpenAI доступний (API key вказано)")
                else:
                    logger.warning("[ImprovedLLM] OpenAI API key не вказано")
                    self._available = False
            elif self.api_type in ["ollama", "local"]:  # 'local' працює як 'ollama'
                try:
                    import requests
                    # Використовуємо вказаний URL або стандартний
                    api_url = self.api_url or "http://localhost:11434"
                    logger.info(f"[ImprovedLLM] Перевірка {'Ollama' if self.api_type == 'ollama' else 'Local'} API: {api_url}")
                    
                    # Перевіряємо доступність API
                    try:
                        response = requests.get(f"{api_url}/api/tags", timeout=5)
                        if response.status_code == 200:
                            # Перевіряємо, чи модель доступна
                            data = response.json()
                            models = [m['name'] for m in data.get('models', [])]
                            logger.info(f"[ImprovedLLM] Доступні моделі: {models}")
                            
                            # Якщо модель не вказана або порожня, використовуємо llama3.2:1b як дефолтну
                            if not self.model or not self.model.strip():
                                self.model = DEFAULT_LLM_MODEL  # Закріплюємо на llama3.2:1b (дуже легка версія)
                                logger.info("[ImprovedLLM] Модель не вказана, використовується llama3.2:1b (закріплено)")
                                print("[ImprovedLLM] Модель не вказана, використовується llama3.2:1b (закріплено)", flush=True)
                            else:
                                logger.info(f"[ImprovedLLM] Використовується модель з конфігурації: {self.model}")
                            
                            # Нормалізуємо назву моделі (прибираємо :latest якщо є)
                            model_name = self.model.replace(':latest', '') if self.model else None
                            
                            # Перевіряємо, чи модель доступна (з :latest або без)
                            model_found = False
                            for available_model in models:
                                available_name = available_model.replace(':latest', '')
                                if model_name and (model_name == available_name or self.model == available_model):
                                    model_found = True
                                    self.model = available_model  # Використовуємо точну назву зі списку
                                    break
                            
                            if model_found:
                                self._available = True
                                logger.info(f"[ImprovedLLM] ✓ Модель {self.model} доступна")
                            else:
                                logger.warning(f"[ImprovedLLM] ✗ Модель {self.model} не знайдена. Доступні: {models}")
                                # Спробуємо знайти легкі моделі (менше 3B параметрів) для обмеженої пам'яті
                                # Пріоритет: llama3.2:1b > llama3.2:3b > інші легкі
                                alternatives = [DEFAULT_LLM_MODEL, 'llama3.2:3b']
                                alternative = None
                                for alt in alternatives:
                                    # Шукаємо точну відповідність або частину назви
                                    matching = [m for m in models if alt.lower() in m.lower() or m.lower().startswith(alt.lower().split(':')[0])]
                                    if matching:
                                        # Віддаємо перевагу точній відповідності
                                        exact = [m for m in matching if alt.lower() in m.lower()]
                                        if exact:
                                            alternative = exact[0]
                                        else:
                                            alternative = matching[0]
                                        break
                                
                                if alternative:
                                    self.model = alternative
                                    self._available = True
                                    logger.info(f"[ImprovedLLM] ✓ Знайдено легку модель: {self.model}")
                                    print(f"[ImprovedLLM] ✓ Знайдено легку модель: {self.model}", flush=True)
                                else:
                                    # Якщо легкі моделі не знайдені, спробуємо будь-яку доступну
                                    if models:
                                        self.model = models[0]
                                        logger.warning(f"[ImprovedLLM] ⚠️ Легкі моделі не знайдені, використовується: {self.model}")
                                        print(f"[ImprovedLLM] ⚠️ Легкі моделі не знайдені, використовується: {self.model}", flush=True)
                                        self._available = True
                                    else:
                                        error_msg = (f"[ImprovedLLM] ✗ Моделі не знайдені. "
                                                   f"Встановіть легку модель: ollama pull {DEFAULT_LLM_MODEL}")
                                        logger.error(error_msg)
                                        print(error_msg, flush=True)
                                        self._available = False
                        else:
                            logger.warning(f"[ImprovedLLM] API повернув статус {response.status_code}")
                            self._available = False
                    except requests.exceptions.ConnectionError as e:
                        logger.warning(f"[ImprovedLLM] ✗ API недоступний (немає з'єднання з {api_url}): {e}")
                        logger.warning("[ImprovedLLM] Переконайтеся, що Ollama запущена: ollama serve")
                        self._available = False
                    except Exception as e:
                        logger.warning(f"[ImprovedLLM] ✗ Помилка перевірки API: {e}")
                        self._available = False
                except Exception as e:
                    logger.warning(f"[ImprovedLLM] ✗ Помилка ініціалізації: {e}")
                    self._available = False
            else:
                logger.warning(f"[ImprovedLLM] Невідомий тип API: {self.api_type}")
                self._available = False
        except Exception as e:
            logger.warning(f"[ImprovedLLM] Помилка перевірки LLM API: {e}")
            self._available = False
    
    def is_available(self) -> bool:
        """Перевірка доступності"""
        return self._available
    
    def correct_text(self, text: str, language: str = "ukrainian", image: Optional[np.ndarray] = None) -> str:
        """
        Виправлення тексту через LLM
        
        Args:
            text: текст з OCR
            language: мова тексту
            
        Returns:
            виправлений текст
        """
        if not self._available or not text or not text.strip():
            return text
        
        # Спочатку застосовуємо просте виправлення для очевидних помилок
        text = self._simple_pre_correction(text, language)
        
        try:
            if self.api_type == "openai":
                return self._correct_with_openai(text, language, image)
            elif self.api_type in ["ollama", "local"]:  # 'local' працює як 'ollama'
                return self._correct_with_ollama(text, language, image)
            else:
                return text
        except Exception as e:
            logger.error(f"Помилка LLM корекції: {e}")
            return text
    
    @staticmethod
    def _simple_pre_correction(text: str, language: str) -> str:
        """
        Просте виправлення очевидних помилок перед ШІ
        
        Args:
            text: текст для виправлення
            language: мова тексту
            
        Returns:
            частково виправлений текст
        """
        if not text or not text.strip():
            return text
        
        corrected = text
        
        # Виправлення для української мови
        if language.lower() in ['ukrainian', 'ukr', 'uk']:
            # "інб" → "їна" (наприклад, "Украінб" → "Україна")
            if 'інб' in corrected:
                corrected = corrected.replace('інб', 'їна')
                logger.info("[ImprovedLLM] Просте виправлення: 'інб' → 'їна'")
            
            # "інo" → "їна" (якщо є "o" замість "а")
            if 'інo' in corrected:
                corrected = corrected.replace('інo', 'їна')
                logger.info("[ImprovedLLM] Просте виправлення: 'інo' → 'їна'")
            
            # "Україноа" → "Україна" (помилка "оа" замість "а")
            if UKRAINA_INCORRECT in corrected:
                corrected = corrected.replace(UKRAINA_INCORRECT, UKRAINA_CORRECT)
                logger.info("[ImprovedLLM] Просте виправлення: '%s' → '%s'",
                          UKRAINA_INCORRECT, UKRAINA_CORRECT)
            if UKRAINA_INCORRECT.lower() in corrected.lower():
                corrected = (corrected.replace(UKRAINA_INCORRECT.lower(), UKRAINA_CORRECT.lower())
                           .replace(UKRAINA_INCORRECT, UKRAINA_CORRECT))
                logger.info("[ImprovedLLM] Просте виправлення: '%s' → '%s'",
                          UKRAINA_INCORRECT.lower(), UKRAINA_CORRECT.lower())
            
            # "Україно." → "Україна" (видалення крапки в кінці)
            if corrected.endswith('Україно.') or corrected.endswith('україно.'):
                corrected = corrected[:-1] + 'а'
                logger.info("[ImprovedLLM] Просте виправлення: 'Україно.' → 'Україна'")
        
        return corrected
    
    def _correct_with_openai(self, text: str, language: str, image: Optional[np.ndarray] = None) -> str:
        """Виправлення через OpenAI API"""
        try:
            import openai  # type: ignore
            
            if not self.api_key:
                return text
            
            client = openai.OpenAI(api_key=self.api_key)
            
            # Конвертуємо зображення в base64, якщо воно є
            image_base64 = None
            if image is not None:
                try:
                    import base64
                    import cv2
                    _, buffer = cv2.imencode('.png', image)
                    image_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                except Exception as e:
                    logger.warning(f"[OpenAI] Не вдалося конвертувати зображення: {e}")
                    image_base64 = None
            
            user_prompt = f"""Мова тексту: {language}

Текст з OCR:
{text}

Виправлений текст:"""
            
            # Формуємо повідомлення з зображенням, якщо воно є
            user_content = []
            if image_base64:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                })
                user_prompt = f"""Мова тексту: {language}

Ти бачиш оригінальне зображення з рукописним текстом. Використай його для виправлення помилок OCR.

Текст з OCR (може містити помилки):
{text}

Виправлений текст (використовуй зображення для перевірки):"""
            
            user_content.append({
                "type": "text",
                "text": user_prompt
            })
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,  # Мінімальна креативність
                max_tokens=2000
            )
            
            corrected = response.choices[0].message.content.strip()
            
            # Видаляємо можливі маркери
            corrected = corrected.replace("```", "").strip()
            
            # Фільтрація небажаних символів (китайські, японські тощо)
            corrected = self._filter_unwanted_characters(corrected, language)
            
            # Витягуємо тільки текст, який відповідає оригіналу (без зайвих додатків)
            corrected = self._extract_only_user_text(corrected, text, language)
            
            # Перевірка на додавання нових символів (як "!")
            if self._has_added_unwanted_chars(corrected, text):
                logger.warning(f"[OpenAI] Результат містить додані небажані символи, використовуємо оригінал: '{text}' -> '{corrected}'")
                return text
            
            # Перевірка на зміну структури (додавання нових слів)
            if self._has_changed_structure_too_much(corrected, text):
                logger.warning(f"[OpenAI] Результат змінив структуру занадто сильно, використовуємо оригінал: '{text}' -> '{corrected}'")
                return text
            
            return corrected if corrected else text
            
        except ImportError:
            logger.warning("OpenAI library не встановлено")
            return text
        except Exception as e:
            logger.error(f"Помилка OpenAI API: {e}")
            return text
    
    def _correct_with_ollama(self, text: str, language: str, image: Optional[np.ndarray] = None) -> str:
        """Виправлення через Ollama API з покращеною обробкою"""
        try:
            import requests
            import time
            
            # Конвертуємо зображення в base64, якщо воно є
            image_base64 = None
            if image is not None:
                try:
                    import base64
                    import cv2
                    _, buffer = cv2.imencode('.png', image)
                    image_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                    logger.info("[Ollama] Зображення конвертовано в base64 для корекції")
                except Exception as e:
                    logger.warning(f"[Ollama] Не вдалося конвертувати зображення: {e}")
                    image_base64 = None
            
            # Покращений промпт для кращого розуміння контексту
            lang_name_uk = "українська" if language == "ukrainian" else "англійська"
            
            image_note = ""
            if image_base64:
                image_note = "\n\nВАЖЛИВО: Ти бачиш оригінальне зображення з рукописним текстом. Використай його для виправлення помилок OCR. Порівняй текст з тим, що написано на зображенні."
            
            user_prompt = f"""Виправ помилки OCR в наступному тексті.

Мова тексту: {lang_name_uk}
{image_note}

Оригінальний текст (може містити помилки OCR):
{text}

Важливо:
- Виправляй ТІЛЬКИ очевидні помилки OCR
- Зберігай оригінальний зміст та структуру
- НЕ додавай нову інформацію
- НЕ перефразовуй
- Поверни ТІЛЬКИ виправлений текст, без пояснень
- Якщо ти бачиш зображення - використай його для перевірки

Виправлений текст:"""
            
            # Формуємо повідомлення для Ollama (підтримка vision моделей)
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
            ]
            
            user_message = {"role": "user", "content": user_prompt}
            if image_base64:
                # Для Ollama vision моделей зображення передається як base64 в images
                user_message["images"] = [image_base64]  # type: ignore
                logger.info("[Ollama] Зображення додано до запиту для корекції")
            
            messages.append(user_message)
            
            # Спробуємо кілька разів з різними налаштуваннями
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    # Використовуємо chat endpoint з системним промптом
                    response = requests.post(
                        f"{self.api_url}/api/chat",
                        json={
                            "model": self.model,
                            "messages": messages,
                            "stream": False,
                            "options": {
                                "temperature": 0.1,  # Мінімальна креативність
                                "top_p": 0.9,
                                "num_predict": 2000,
                                "repeat_penalty": 1.1  # Зменшує повторення
                            }
                        },
                        timeout=90  # Збільшено таймаут
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Обробка різних форматів відповіді Ollama
                        corrected = ""
                        if "message" in result:
                            if isinstance(result["message"], dict):
                                corrected = result["message"].get("content", "").strip()
                            else:
                                corrected = str(result["message"]).strip()
                        elif "response" in result:
                            corrected = result.get("response", "").strip()
                        else:
                            # Спробуємо знайти текст в будь-якому полі
                            corrected = str(result).strip()
                        
                        # Очищення від маркерів та зайвих символів
                        corrected = corrected.replace("```", "").strip()
                        corrected = corrected.replace("```text", "").strip()
                        corrected = corrected.replace("```markdown", "").strip()
                        corrected = corrected.replace("Виправлений текст:", "").strip()
                        corrected = corrected.replace("Corrected text:", "").strip()
                        
                        # Видаляємо весь промпт, якщо модель повернула його
                        # Шукаємо рядок "Оригінальний текст" і беремо текст після нього
                        if "Оригінальний текст" in corrected or "Original text" in corrected:
                            # Спробуємо витягти тільки виправлений текст
                            lines = corrected.split('\n')
                            result_lines = []
                            skip_until_text = False
                            for line in lines:
                                # Пропускаємо всі рядки до оригінального тексту
                                if "Оригінальний текст" in line or "Original text" in line:
                                    skip_until_text = True
                                    continue
                                # Пропускаємо рядки з правилами та інструкціями
                                if skip_until_text and ("ПРАВИЛА" in line or "RULES" in line or "ТВОЄ ЄДИНЕ ЗАВДАННЯ" in line or "YOUR ONLY TASK" in line):
                                    continue
                                if skip_until_text and line.strip() and not line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '-', '*')):
                                    result_lines.append(line)
                            if result_lines:
                                corrected = '\n'.join(result_lines).strip()
                                logger.info(f"[Ollama] Витягнуто текст з промпту: '{corrected[:50]}...'")
                        
                        # Видаляємо початкові/кінцеві лапки якщо є
                        if corrected.startswith('"') and corrected.endswith('"'):
                            corrected = corrected[1:-1]
                        if corrected.startswith("'") and corrected.endswith("'"):
                            corrected = corrected[1:-1]
                        
                        # Якщо результат все ще містить багато тексту з промпту, спробуємо витягти тільки перший рядок
                        if len(corrected) > len(text) * 2 and '\n' in corrected:
                            first_line = corrected.split('\n')[0].strip()
                            # Якщо перший рядок виглядає як виправлений текст (не містить інструкцій)
                            if first_line and len(first_line) < 100 and not any(word in first_line for word in ['ПРАВИЛА', 'RULES', 'ЗАВДАННЯ', 'TASK', 'Мова тексту', 'Text language']):
                                corrected = first_line
                                logger.info(f"[Ollama] Використано перший рядок як виправлений текст: '{corrected}'")
                        
                        # Фільтрація небажаних символів (китайські, японські тощо)
                        corrected = self._filter_unwanted_characters(corrected, language)
                        
                        # Витягуємо тільки текст, який відповідає оригіналу (без зайвих додатків)
                        corrected = self._extract_only_user_text(corrected, text, language)
                        
                        # Перевірка, чи результат не порожній та не дуже відрізняється від оригіналу
                        if corrected and len(corrected) > 0:
                            # Перевіряємо якість результату - якщо він виглядає краще, приймаємо його
                            quality_score = self._assess_text_quality(corrected, text, language)
                            
                            # Якщо якість покращилася, приймаємо результат навіть з більшими змінами
                            if quality_score > 0.6:
                                logger.info(f"[Ollama] Якість результату покращилася (score: {quality_score:.2f}), приймаємо: '{text[:50]}...' -> '{corrected[:50]}...'")
                                return corrected
                            
                            # Перевірка на додавання нових символів (як "!") - тільки якщо якість не покращилася
                            if quality_score <= 0.3 and self._has_added_unwanted_chars(corrected, text):
                                logger.warning(f"[Ollama] Результат містить додані небажані символи та не покращив якість, використовуємо оригінал: '{text}' -> '{corrected}'")
                                return text
                            
                            # Перевірка на зміну структури - тільки для явно поганих результатів
                            if quality_score <= 0.2 and self._has_changed_structure_too_much(corrected, text):
                                logger.warning(f"[Ollama] Результат змінив структуру занадто сильно та не покращив якість, використовуємо оригінал: '{text}' -> '{corrected}'")
                                return text
                            
                            original_len = len(text)
                            corrected_len = len(corrected)
                            length_ratio = corrected_len / max(original_len, 1)
                            
                            # Більш гнучкі межі для довжини - дозволяємо більші зміни
                            if original_len < 10:
                                # Для коротких слів дозволяємо від 0.3 до 2.5 (більш гнучко)
                                min_ratio = 0.3
                                max_ratio = 2.5
                            else:
                                # Для довгих текстів також більш гнучко
                                min_ratio = 0.2
                                max_ratio = 3.0
                            
                            if min_ratio <= length_ratio <= max_ratio:
                                logger.info(f"[Ollama] Успішно виправлено текст: '{text[:50]}...' -> '{corrected[:50]}...' (довжина: {original_len} -> {corrected_len}, ratio: {length_ratio:.2f})")
                                return corrected
                            else:
                                # Якщо довжина виходить за межі, але якість покращилася, все одно приймаємо
                                if quality_score > 0.4:
                                    logger.info(f"[Ollama] Довжина виходить за межі, але якість покращилася (score: {quality_score:.2f}), приймаємо результат")
                                    return corrected
                                logger.warning(f"[Ollama] Результат підозрілий (довжина {corrected_len} vs {original_len}, ratio: {length_ratio:.2f}), використовуємо оригінал")
                                return text
                        else:
                            logger.warning("[Ollama] Порожній результат, використовуємо оригінал")
                            return text
                    else:
                        error_text = response.text[:200] if hasattr(response, 'text') else "Unknown error"
                        logger.warning(f"[Ollama] Помилка HTTP {response.status_code}: {error_text}")
                        if attempt < max_retries - 1:
                            time.sleep(1)  # Невелика затримка перед повторною спробою
                            continue
                        return text
                        
                except requests.exceptions.Timeout:
                    logger.warning(f"[Ollama] Таймаут (спроба {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return text
                except requests.exceptions.ConnectionError as e:
                    logger.error(f"[Ollama] Помилка з'єднання: {e}")
                    return text
                except Exception as e:
                    logger.error(f"[Ollama] Помилка на спробі {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    return text
            
            return text
                
        except Exception as e:
            logger.error(f"Помилка Ollama API: {e}", exc_info=True)
            return text
    
    @staticmethod
    def _has_added_unwanted_chars(corrected: str, original: str) -> bool:
        """
        Перевірка, чи ШІ додав небажані символи (як "!", "?" в кінці, якщо їх не було)
        
        Args:
            corrected: виправлений текст
            original: оригінальний текст
            
        Returns:
            True, якщо додані небажані символи
        """
        # Символи, які не повинні з'являтися, якщо їх не було в оригіналі
        unwanted_chars = ['!', '?']
        
        for char in unwanted_chars:
            original_count = original.count(char)
            corrected_count = corrected.count(char)
            if corrected_count > original_count:
                return True
        
        return False
    
    @staticmethod
    def _assess_text_quality(corrected: str, original: str, language: str) -> float:
        """
        Оцінка якості виправленого тексту (0.0 - 1.0)
        
        Args:
            corrected: виправлений текст
            original: оригінальний текст
            language: мова тексту
            
        Returns:
            Оцінка якості (0.0 = погано, 1.0 = відмінно)
        """
        if not corrected or not original:
            return 0.0
        
        score = 0.5  # Базовий бал
        
        if language.lower() in ['ukrainian', 'ukr', 'uk']:
            # Перевіряємо наявність кирилиці
            has_cyrillic = any('\u0400' <= char <= '\u04FF' for char in corrected)
            original_has_cyrillic = any('\u0400' <= char <= '\u04FF' for char in original)
            
            if has_cyrillic and not original_has_cyrillic:
                score += 0.3  # Додали кирилицю - це добре
            
            # Перевіряємо наявність правильних українських слів
            common_ukrainian_words = ['привіт', 'україна', 'любов', 'сон', 'цей', 'той', 'це', 'так', 'ні', 'добрий', 'день', 'вечір', 'ранок']
            corrected_lower = corrected.lower()
            original_lower = original.lower()
            
            corrected_words_found = sum(1 for word in common_ukrainian_words if word in corrected_lower)
            original_words_found = sum(1 for word in common_ukrainian_words if word in original_lower)
            
            if corrected_words_found > original_words_found:
                score += 0.2  # Знайшли більше правильних слів
            
            # Перевіряємо, чи текст виглядає як правильне українське слово
            if has_cyrillic and len(corrected.strip()) >= 3:
                # Якщо текст містить кирилицю і виглядає як слово, це добре
                if all('\u0400' <= char <= '\u04FF' or char.isdigit() or char in ' .,;:!?-' for char in corrected):
                    score += 0.1
        
        # Перевіряємо, чи текст не містить очевидних помилок
        if 'Тіль:' in corrected or 'Body:' in corrected:
            score -= 0.3  # Містить небажані префікси
        
        # Перевіряємо, чи текст не містить китайських символів
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in corrected)
        if has_chinese:
            score -= 0.5  # Містить китайські символи
        
        return max(0.0, min(1.0, score))  # Обмежуємо від 0.0 до 1.0
    
    @staticmethod
    def _has_changed_structure_too_much(corrected: str, original: str) -> bool:
        """
        Перевірка, чи ШІ змінив структуру тексту занадто сильно (додав нові слова, змінив порядок)
        
        Args:
            corrected: виправлений текст
            original: оригінальний текст
            
        Returns:
            True, якщо структура змінена занадто сильно
        """
        # Порівнюємо кількість слів
        original_words = original.split()
        corrected_words = corrected.split()
        
        # Якщо кількість слів змінилася більш ніж на 100%, це підозріло (було 50%)
        if len(original_words) > 0:
            word_ratio = len(corrected_words) / len(original_words)
            if word_ratio < 0.3 or word_ratio > 2.0:  # Було 0.5-1.5, тепер 0.3-2.0
                return True
        
        # Для коротких текстів (1-2 слова) перевіряємо схожість символів
        # Але тільки якщо зміни дуже радикальні
        if len(original_words) <= 2:
            original_lower = original.lower().strip()
            corrected_lower = corrected.lower().strip()
            
            # Для коротких текстів перевіряємо схожість символів
            # Якщо менше 30% символів співпадають, це підозріло (було 50%)
            if len(original_lower) > 0:
                matching_chars = sum(1 for c in original_lower if c in corrected_lower)
                similarity = matching_chars / len(original_lower)
                if similarity < 0.3:  # Було 0.5, тепер 0.3
                    return True
        
        return False
    
    @staticmethod
    def _extract_only_user_text(corrected: str, original: str, language: str) -> str:
        """
        Витягує тільки текст користувача, видаляючи всі зайві додатки від ШІ
        
        Args:
            corrected: виправлений текст від ШІ
            original: оригінальний текст від OCR
            language: мова тексту
            
        Returns:
            тільки текст користувача без зайвих додатків
        """
        if not corrected or not original:
            return corrected if corrected else original
        
        # Видаляємо всі рядки, які містять ключові слова з промпту
        forbidden_phrases = [
            'ПРАВИЛА', 'RULES', 'ЗАВДАННЯ', 'TASK', 'Мова тексту', 'Text language',
            'Оригінальний текст', 'Original text', 'Виправлений текст', 'Corrected text',
            'ТВОЄ ЄДИНЕ ЗАВДАННЯ', 'YOUR ONLY TASK', 'КРИТИЧНО ВАЖЛИВО', 'CRITICALLY IMPORTANT',
            'ПРИКЛАДИ', 'EXAMPLES', 'ФОРМАТ ВІДПОВІДІ', 'RESPONSE FORMAT',
            'НЕ відповідай', 'НЕ додавай', 'НЕ видаляй', 'НЕ перефразовуй', 'НЕ пиши',
            'DON\'T answer', 'DON\'T add', 'DON\'T remove', 'DON\'T rephrase', 'DON\'T write',
            'Виправляй ТІЛЬКИ', 'Correct ONLY', 'Зберігай', 'Keep', 'Використовуй ТІЛЬКИ',
            'Use ONLY', 'ВИДАЛЯЙ', 'REMOVE', 'Тіль:', 'Тіль', 'Body:', 'Body'
        ]
        
        lines = corrected.split('\n')
        filtered_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Пропускаємо рядки з забороненими фразами
            if any(phrase in line_stripped for phrase in forbidden_phrases):
                continue
            
            # Пропускаємо рядки, які виглядають як інструкції (починаються з цифр, дефісів, зірочок)
            if line_stripped.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '-', '*', '•')):
                continue
            
            # Пропускаємо рядки, які містять тільки пунктуацію або спецсимволи
            if all(c in '.,;:!?\-()[]"\'«»—…/\n\r\t ' for c in line_stripped):
                continue
            
            filtered_lines.append(line)
        
        if filtered_lines:
            result = '\n'.join(filtered_lines).strip()
        else:
            # Якщо всі рядки відфільтровані, спробуємо витягти тільки перший осмислений рядок
            for line in lines:
                line_stripped = line.strip()
                if line_stripped and len(line_stripped) > 2:
                    # Перевіряємо, чи рядок не містить заборонених фраз
                    if not any(phrase in line_stripped for phrase in forbidden_phrases):
                        result = line_stripped
                        break
            else:
                result = corrected
        
        # Якщо результат значно довший за оригінал, спробуємо знайти найбільш схожу частину
        if len(result) > len(original) * 1.5:
            # Спробуємо знайти найбільш схожу частину, порівнюючи слова
            original_words = original.split()
            result_words = result.split()
            
            # Шукаємо найбільшу послідовність слів, яка відповідає оригіналу
            best_match = result
            best_match_length = 0
            
            # Перевіряємо перші N слів, де N = кількість слів в оригіналі * 1.2
            max_words = int(len(original_words) * 1.2)
            if len(result_words) > max_words:
                # Беремо перші N слів
                candidate = ' '.join(result_words[:max_words])
                if len(candidate) <= len(original) * 1.3:
                    best_match = candidate
                    best_match_length = len(candidate)
            
            # Також перевіряємо перший рядок
            first_line = result.split('\n')[0].strip()
            first_line_words = first_line.split()
            if len(first_line_words) <= max_words and len(first_line) <= len(original) * 1.3:
                if len(first_line) > best_match_length:
                    best_match = first_line
                    best_match_length = len(first_line)
            
            # Якщо знайшли кращий варіант, використовуємо його
            if best_match_length > 0:
                result = best_match
            else:
                # Якщо нічого не знайшли, обрізаємо до розумної довжини
                max_length = int(len(original) * 1.2)
                result = result[:max_length].strip()
                # Обрізаємо на останньому пробілі, щоб не обрізати слово
                last_space = result.rfind(' ')
                if last_space > len(original) * 0.5:
                    result = result[:last_space].strip()
        
        # Видаляємо префікси типу "Тіль:", "Body:" на початку тексту
        prefixes_to_remove = ['Тіль:', 'Тіль', 'Body:', 'Body', 'Текст:', 'Text:']
        for prefix in prefixes_to_remove:
            if result.startswith(prefix):
                result = result[len(prefix):].strip()
                # Якщо після префіксу є двокрапка, видаляємо його теж
                if result.startswith(':'):
                    result = result[1:].strip()
                logger.info(f"[ImprovedLLM] Видалено префікс '{prefix}' з результату")
        
        # Фінальна перевірка - якщо результат все ще містить заборонені фрази, повертаємо оригінал
        if any(phrase in result for phrase in forbidden_phrases):
            logger.warning(f"[ImprovedLLM] Результат містить заборонені фрази, використовуємо оригінал")
            return original
        
        return result
    
    @staticmethod
    def _filter_unwanted_characters(text: str, language: str) -> str:
        """
        Фільтрація небажаних символів (китайські, японські тощо)
        
        Args:
            text: текст для фільтрації
            language: мова тексту
            
        Returns:
            відфільтрований текст
        """
        if not text:
            return text
        
        # Дозволені символи для української мови
        if language.lower() in ['ukrainian', 'ukr', 'uk']:
            # Кирилиця (включаючи всі українські літери), латиниця (для деяких слів), цифри, пунктуація, пробіли
            # Використовуємо Unicode діапазони для кращої підтримки
            filtered_chars = []
            for char in text:
                # Дозволяємо кирилицю (Unicode діапазон)
                if '\u0400' <= char <= '\u04FF':  # Кирилиця
                    filtered_chars.append(char)
                # Дозволяємо латиницю
                elif 'a' <= char <= 'z' or 'A' <= char <= 'Z':
                    filtered_chars.append(char)
                # Дозволяємо цифри
                elif '0' <= char <= '9':
                    filtered_chars.append(char)
                # Дозволяємо пунктуацію та пробіли
                elif char in ' .,;:!?\-()[]"\'«»—…/\n\r\t':
                    filtered_chars.append(char)
                # Всі інші символи (китайські, японські тощо) - видаляємо
        else:
            # Для англійської - тільки латиниця, цифри, пунктуація
            filtered_chars = []
            for char in text:
                if 'a' <= char <= 'z' or 'A' <= char <= 'Z':
                    filtered_chars.append(char)
                elif '0' <= char <= '9':
                    filtered_chars.append(char)
                elif char in ' .,;:!?\-()[]"\'/\n\r\t':
                    filtered_chars.append(char)
        
        filtered_text = ''.join(filtered_chars)
        
        # Якщо після фільтрації текст значно змінився, логуємо
        if len(filtered_text) < len(text) * 0.8:
            logger.warning(f"[ImprovedLLM] Видалено небажані символи: '{text[:50]}...' -> '{filtered_text[:50]}...'")
        elif filtered_text != text:
            logger.info(f"[ImprovedLLM] Видалено небажані символи з тексту")
        
        return filtered_text if filtered_text else text
    
    @staticmethod
    def simple_correction(text: str) -> str:
        """Швидке виправлення без LLM"""
        if not text:
            return text
        
        import re
        
        # Нормалізація пробілів
        text = ' '.join(text.split())
        
        # Виправлення поширених OCR помилок
        text = re.sub(r'\brn\b', 'm', text)
        text = re.sub(r'vv+', 'w', text)
        
        # Українські виправлення
        corrections = {
            'т0': 'то',
            'н0': 'но',
            'в0': 'во',
        }
        
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        
        return text
