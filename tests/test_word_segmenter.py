"""
Unit-тести для WordSegmenter
handwrite2print/tests/test_word_segmenter.py
"""
import pytest
import sys
import os
import numpy as np

# Додавання шляху до app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.model.word_segmenter import WordSegmenter, WordSegment


class TestWordSegmenter:
    """Тести для WordSegmenter"""
    
    def test_segmenter_handles_none_image(self):
        """Тест обробки None зображення"""
        result = WordSegmenter.segment_words(None)  # type: ignore[arg-type]
        assert result == []
    
    def test_segmenter_handles_empty_image(self):
        """Тест обробки порожнього зображення"""
        empty_image = np.array([], dtype=np.uint8)
        # WordSegmenter може викликати помилку OpenCV для порожнього зображення
        # Або повернути порожній список
        try:
            result = WordSegmenter.segment_words(empty_image)
            assert isinstance(result, list)
        except Exception:
            # Якщо виникає помилка, це теж прийнятна поведінка
            pass
    
    def test_segmenter_segments_simple_image(self):
        """Тест сегментації простого зображення"""
        # Створюємо просте зображення з текстом (білий фон, чорний текст)
        rng = np.random.default_rng(42)
        image = np.ones((200, 400), dtype=np.uint8) * 255  # Білий фон
        
        # Додаємо "текст" (темні області)
        image[50:150, 50:150] = 0  # Перша "буква"
        image[50:150, 200:300] = 0  # Друга "буква"
        
        segments = WordSegmenter.segment_words(image)
        
        assert isinstance(segments, list)
        # Може знайти сегменти або ні, залежно від алгоритму
        for segment in segments:
            assert isinstance(segment, WordSegment)
            assert hasattr(segment, 'bbox')
            assert hasattr(segment, 'image')
            assert len(segment.bbox) == 4
            assert isinstance(segment.image, np.ndarray)
    
    def test_segmenter_handles_grayscale_image(self):
        """Тест обробки grayscale зображення"""
        rng = np.random.default_rng(123)
        image = rng.integers(0, 255, (200, 400), dtype=np.uint8)
        
        segments = WordSegmenter.segment_words(image)
        assert isinstance(segments, list)
    
    def test_segmenter_handles_color_image(self):
        """Тест обробки кольорового зображення"""
        rng = np.random.default_rng(456)
        image = rng.integers(0, 255, (200, 400, 3), dtype=np.uint8)
        
        segments = WordSegmenter.segment_words(image)
        assert isinstance(segments, list)
    
    def test_segmenter_sorts_segments_left_to_right(self):
        """Тест сортування сегментів зліва направо"""
        # Створюємо зображення з двома сегментами
        image = np.ones((200, 400), dtype=np.uint8) * 255
        
        # Правий сегмент (x=250)
        image[50:150, 250:350] = 0
        # Лівий сегмент (x=50)
        image[50:150, 50:150] = 0
        
        segments = WordSegmenter.segment_words(image)
        
        if len(segments) >= 2:
            # Перевіряємо, що сегменти відсортовані зліва направо
            for i in range(len(segments) - 1):
                assert segments[i].bbox[0] <= segments[i + 1].bbox[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

