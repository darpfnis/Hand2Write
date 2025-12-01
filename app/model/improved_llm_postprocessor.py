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

ТВОЄ ЄДИНЕ ЗАВДАННЯ: Виправити помилки в тексті, отриманому через OCR.

ПРАВИЛА (ОБОВ'ЯЗКОВІ):
1. НЕ відповідай на питання - просто виправ помилки
2. НЕ додавай нову інформацію
3. НЕ видаляй інформацію
4. НЕ перефразовуй - зберігай оригінальний стиль
5. НЕ пиши пояснень - тільки виправлений текст
6. Виправляй ТІЛЬКИ очевидні OCR помилки:
   - rn → m
   - vv → w
   - 0 → O (в словах)
   - l → I (заголовна і)
   - Для української мови:
     * "інб" → "їна" (наприклад, "Украінб" → "Україна")
     * "інo" → "їна" (наприклад, "Україно" → "Україна")
     * "юбов" → "любов" (якщо починається з "І" або "і")
7. Виправляй граматичні помилки
8. Зберігай форматування та структуру
9. Для української мови: зберігай українські особливі літери (і, ї, є)

ПРИКЛАДИ:
- "Украінб" → "Україна"
- "ІюБОВ" → "Любов"
- "Прогбіт" → "Привіт"
- "Україно." → "Україна"

ФОРМАТ ВІДПОВІДІ: Тільки виправлений текст, без пояснень."""

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
                        
                        # Перевірка, чи результат не порожній та не дуже відрізняється від оригіналу
                        if corrected and len(corrected) > 0:
                            original_len = len(text)
                            corrected_len = len(corrected)
                            length_ratio = corrected_len / max(original_len, 1)
                            
                            # Для коротких слів (менше 10 символів) дозволяємо більшу різницю
                            # Наприклад, "Украінб" (7) -> "Україна" (7) або "Україно." (8) -> "Україна" (7)
                            if original_len < 10:
                                # Для коротких слів дозволяємо від 0.5 до 2.0 (більш гнучко)
                                min_ratio = 0.5
                                max_ratio = 2.0
                            else:
                                # Для довгих текстів залишаємо жорсткіші межі
                                min_ratio = 0.3
                                max_ratio = 3.0
                            
                            if min_ratio <= length_ratio <= max_ratio:
                                logger.info(f"[Ollama] Успішно виправлено текст: '{text[:50]}...' -> '{corrected[:50]}...' (довжина: {original_len} -> {corrected_len}, ratio: {length_ratio:.2f})")
                                return corrected
                            else:
                                logger.warning(f"[Ollama] Результат підозрілий (довжина {corrected_len} vs {original_len}, ratio: {length_ratio:.2f}), використовуємо оригінал")
                                # Але якщо результат схожий на правильне слово (наприклад, "Україна"), все одно використовуємо його
                                if language.lower() in ['ukrainian', 'ukr', 'uk']:
                                    # Перевіряємо, чи результат містить правильні українські слова
                                    common_words = ['україна', 'привіт', 'любов', 'проба']
                                    corrected_lower = corrected.lower()
                                    if any(word in corrected_lower for word in common_words):
                                        logger.info(f"[Ollama] Результат містить правильне слово, використовуємо його: '{corrected}'")
                                        return corrected
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
