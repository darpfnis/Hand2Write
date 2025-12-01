"""
LLM Post-processing для виправлення помилок OCR
handwrite2print/app/model/llm_postprocessor.py
"""
import logging
from typing import Optional, Dict, Any
import json

logger = logging.getLogger(__name__)


class LLMPostProcessor:
    """Клас для пост-обробки тексту через LLM API"""
    
    def __init__(self, api_type: str = "openai", api_key: Optional[str] = None, 
                 api_url: Optional[str] = None):
        """
        Ініціалізація LLM пост-процесора
        
        Args:
            api_type: тип API ('openai', 'local', 'ollama')
            api_key: API ключ (для OpenAI)
            api_url: URL локального API (для local/ollama)
        """
        self.api_type = api_type
        self.api_key = api_key
        self.api_url = api_url or "http://localhost:11434"  # Default Ollama
        self._available = False
        self._check_availability()
    
    def _check_availability(self):
        """Перевірка доступності API"""
        try:
            if self.api_type == "openai":
                if self.api_key:
                    self._available = True
                else:
                    logger.warning("OpenAI API key не вказано")
            elif self.api_type in ["local", "ollama"]:
                # Перевірка доступності локального API
                try:
                    import requests
                    response = requests.get(f"{self.api_url}/api/tags", timeout=2)
                    if response.status_code == 200:
                        self._available = True
                except Exception:
                    logger.warning(f"Локальний LLM API недоступний: {self.api_url}")
            else:
                self._available = False
        except Exception as e:
            logger.warning(f"Помилка перевірки LLM API: {e}")
            self._available = False
    
    def is_available(self) -> bool:
        """Перевірка доступності"""
        return self._available
    
    def correct_text(self, text: str, language: str = "ukrainian", 
                     context: Optional[str] = None) -> str:
        """
        Виправлення тексту через LLM з розумною перевіркою
        
        Args:
            text: текст з OCR (може містити помилки)
            language: мова тексту
            context: додатковий контекст для кращого виправлення
            
        Returns:
            виправлений текст
        """
        if not self._available or not text or not text.strip():
            return text
        
        # Спочатку застосовуємо просте виправлення
        text = self.simple_correction(text)
        
        try:
            if self.api_type == "openai":
                return self._correct_with_openai(text, language, context)
            elif self.api_type == "ollama":
                return self._correct_with_ollama(text, language, context)
            elif self.api_type == "local":
                return self._correct_with_local(text, language, context)
            else:
                return text
        except Exception as e:
            logger.error(f"Помилка LLM пост-обробки: {e}")
            return text  # Повертаємо оригінальний текст при помилці
    
    def validate_and_correct(self, text: str, language: str = "ukrainian") -> Dict[str, Any]:
        """
        Валідація та виправлення тексту з детальною інформацією
        
        Args:
            text: текст з OCR
            language: мова тексту
            
        Returns:
            словник з результатами: {
                'original': оригінальний текст,
                'corrected': виправлений текст,
                'confidence': впевненість (0-1),
                'changes': список змін,
                'is_valid': чи текст валідний
            }
        """
        result = {
            'original': text,
            'corrected': text,
            'confidence': 0.5,
            'changes': [],
            'is_valid': True
        }
        
        if not text or not text.strip():
            result['is_valid'] = False
            return result
        
        # Просте виправлення
        corrected = self.simple_correction(text)
        if corrected != text:
            result['changes'].append("Просте виправлення застосовано")
            result['corrected'] = corrected
            result['confidence'] = 0.6
        
        # LLM виправлення (якщо доступно)
        if self._available:
            try:
                llm_corrected = self.correct_text(corrected, language)
                if llm_corrected != corrected:
                    result['changes'].append("LLM виправлення застосовано")
                    result['corrected'] = llm_corrected
                    result['confidence'] = 0.9
            except Exception as e:
                logger.warning(f"LLM валідація не вдалася: {e}")
        
        # Перевірка валідності
        result['is_valid'] = len(result['corrected'].strip()) > 0
        
        return result
    
    def _correct_with_openai(self, text: str, language: str, context: Optional[str] = None) -> str:
        """Виправлення через OpenAI API з покращеним промптом"""
        try:
            import openai  # type: ignore
            
            if not self.api_key:
                return text
            
            client = openai.OpenAI(api_key=self.api_key)
            
            # Покращений промпт для кращого виправлення OCR помилок
            system_prompt = """Ти експерт з виправлення текстів, отриманих через OCR (оптичне розпізнавання символів).
Твоя задача:
1. Виправляти типові OCR помилки (наприклад: rn→m, vv→w, 0→O, l→I)
2. Виправляти граматичні помилки
3. Відновлювати правильні слова з контексту
4. Зберігати оригінальний зміст та структуру тексту
5. Не додавати нову інформацію, якої немає в оригіналі
6. Зберігати форматування (переноси рядків, пробіли)

Повертай ТІЛЬКИ виправлений текст, без пояснень."""
            
            user_prompt = f"""Виправ помилки OCR та граматичні помилки в наступному тексті.
Мова тексту: {language}"""
            
            if context:
                user_prompt += f"\nКонтекст: {context}"
            
            user_prompt += f"""

Текст для виправлення:
{text}

Виправлений текст:"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Використовуємо більш точну модель
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  # Нижча температура для більшої точності
                max_tokens=2000
            )
            
            corrected = response.choices[0].message.content.strip()
            
            # Видаляємо можливі маркери форматування
            corrected = corrected.replace("```", "").strip()
            if corrected.startswith("Виправлений текст:"):
                corrected = corrected.replace("Виправлений текст:", "").strip()
            
            return corrected if corrected else text
            
        except ImportError:
            logger.warning("OpenAI library не встановлено. Встановіть: pip install openai")
            return text
        except Exception as e:
            logger.error(f"Помилка OpenAI API: {e}")
            return text
    
    def _correct_with_ollama(self, text: str, language: str, context: Optional[str] = None) -> str:
        """Виправлення через Ollama API з покращеним промптом"""
        try:
            import requests
            
            system_prompt = "Ти експерт з виправлення текстів OCR. Виправляй помилки, зберігаючи зміст."
            
            prompt = f"""Виправ OCR помилки та граматичні помилки в тексті.
Мова: {language}"""
            
            if context:
                prompt += f"\nКонтекст: {context}"
            
            prompt += f"""

Текст:
{text}

Виправлений текст:"""
            
            # Використовуємо новий формат API Ollama (chat)
            response = requests.post(
                f"{self.api_url}/api/chat",
                json={
                    "model": "llama3.2",  # Оновлена модель
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.2
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                # Новий формат API повертає повідомлення в messages
                if "message" in result:
                    corrected = result["message"].get("content", "").strip()
                else:
                    corrected = result.get("response", "").strip()
                
                # Видаляємо можливі маркери форматування
                corrected = corrected.replace("```", "").strip()
                return corrected if corrected else text
            else:
                # Fallback на старий формат API
                response = requests.post(
                    f"{self.api_url}/api/generate",
                    json={
                        "model": "llama3.2",
                        "prompt": f"{system_prompt}\n\n{prompt}",
                        "stream": False
                    },
                    timeout=60
                )
                if response.status_code == 200:
                    result = response.json()
                    corrected = result.get("response", "").strip()
                    return corrected if corrected else text
                return text
                
        except ImportError:
            logger.warning("requests library не встановлено")
            return text
        except Exception as e:
            logger.error(f"Помилка Ollama API: {e}")
            return text
    
    def _correct_with_local(self, text: str, language: str, context: Optional[str] = None) -> str:
        """Виправлення через локальний API (загальний формат)"""
        try:
            import requests
            
            prompt = f"""Виправ OCR помилки та граматичні помилки в тексті. Мова: {language}"""
            
            if context:
                prompt += f"\nКонтекст: {context}"
            
            prompt += f"\n\nТекст:\n{text}\n\nВиправлений текст:"
            
            response = requests.post(
                self.api_url,
                json={
                    "prompt": prompt, 
                    "language": language,
                    "system": "Ти експерт з виправлення текстів OCR. Виправляй помилки, зберігаючи зміст."
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                corrected = result.get("text", result.get("response", result.get("corrected", text))).strip()
                return corrected if corrected else text
            else:
                return text
                
        except Exception as e:
            logger.error(f"Помилка локального API: {e}")
            return text
    
    @staticmethod
    def simple_correction(text: str) -> str:
        """
        Просте виправлення без LLM (базові правила та евристики)
        
        Args:
            text: текст для виправлення
            
        Returns:
            частково виправлений текст
        """
        if not text:
            return text
        
        # Видалення подвійних пробілів та нормалізація пробілів
        text = ' '.join(text.split())
        
        # Виправлення поширених OCR помилок (тільки в контексті слів)
        import re
        
        # Виправлення rn → m (тільки якщо це не частина слова "rn")
        # Обережно: не замінюємо "rn" в словах типу "burn", "turn"
        text = re.sub(r'\brn\b', 'm', text)
        
        # Виправлення подвійних символів
        text = re.sub(r'vv+', 'w', text)
        text = re.sub(r'll+', 'll', text)  # Залишаємо подвійні ll
        
        # Виправлення цифр в тексті (0 → O тільки в контексті букв)
        # Це складніше, тому пропускаємо автоматичне виправлення
        
        # Видалення артефактів OCR (одиночні символи, які не є словами)
        # Залишаємо як є, бо це може бути частиною тексту
        
        # Виправлення поширених помилок для української мови
        ukrainian_corrections = {
            'т0': 'то',
            'н0': 'но',
            'в0': 'во',
        }
        
        for wrong, correct in ukrainian_corrections.items():
            text = text.replace(wrong, correct)
        
        return text

