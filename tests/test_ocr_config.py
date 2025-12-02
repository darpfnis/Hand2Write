"""
Unit-тести для OCRConfig
handwrite2print/tests/test_ocr_config.py
"""
import pytest
import sys
import os
import json
from pathlib import Path
from tempfile import TemporaryDirectory

# Додавання шляху до app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.model.ocr_config import OCRConfig


class TestOCRConfig:
    """Тести для OCRConfig"""
    
    def test_config_default_values(self):
        """Тест дефолтних значень конфігурації"""
        config = OCRConfig()
        
        assert config.config is not None
        assert isinstance(config.config, dict)
    
    def test_config_loads_from_file(self, tmp_path):
        """Тест завантаження конфігурації з файлу"""
        config_file = tmp_path / "test_config.json"
        
        test_config = {
            "preprocessing": {
                "binarization": False,
                "denoise": True
            },
            "llm": {
                "api_type": "openai",
                "model": "gpt-4"
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(test_config, f)
        
        config = OCRConfig(config_file=config_file)
        
        assert config.config_file == config_file
        assert config.config == test_config
    
    def test_config_handles_missing_file(self):
        """Тест обробки відсутнього файлу конфігурації"""
        non_existent_file = Path("/nonexistent/path/config.json")
        config = OCRConfig(config_file=non_existent_file)
        
        # Має використати дефолтні значення
        assert config.config is not None
    
    def test_config_handles_invalid_json(self, tmp_path):
        """Тест обробки невалідного JSON"""
        config_file = tmp_path / "invalid_config.json"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write("invalid json content {")
        
        # Має обробити помилку та використати дефолтні значення
        config = OCRConfig(config_file=config_file)
        assert config.config is not None
    
    def test_get_llm_config_returns_dict(self):
        """Тест, що get_llm_config повертає словник"""
        config_obj = OCRConfig()
        llm_config = config_obj.get_llm_config()
        
        assert isinstance(llm_config, dict)
    
    def test_get_llm_config_has_required_keys(self):
        """Тест наявності обов'язкових ключів"""
        config_obj = OCRConfig()
        llm_config = config_obj.get_llm_config()
        
        # Перевіряємо, що є основні ключі
        assert 'api_type' in llm_config or 'model' in llm_config or 'enabled' in llm_config
    
    def test_get_preprocessing_config(self):
        """Тест отримання конфігурації препроцесингу"""
        config_obj = OCRConfig()
        preprocessing_config = config_obj.get_preprocessing_config()
        
        assert isinstance(preprocessing_config, dict)
    
    def test_get_default_engine(self):
        """Тест отримання дефолтного рушія"""
        config_obj = OCRConfig()
        engine = config_obj.get_default_engine()
        
        assert isinstance(engine, str)
        assert engine in ['tesseract', 'easyocr', 'paddleocr']
    
    def test_is_fallback_enabled(self):
        """Тест перевірки fallback"""
        config_obj = OCRConfig()
        fallback = config_obj.is_fallback_enabled()
        
        assert isinstance(fallback, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

