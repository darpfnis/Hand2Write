"""
Віджет полотна для ручного введення тексту
handwrite2print/app/view/canvas_widget.py
"""
from typing import Optional
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QPainter, QPen, QImage, QColor
import tempfile
import os


class CanvasWidget(QWidget):
    """Віджет для малювання рукописного тексту"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 400)
        self.image = QImage(self.size(), QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.white)
        
        self.drawing = False
        self.last_point = QPoint()
        
        self.pen_color = QColor(0, 0, 0)
        self.pen_width = 4
        self.current_tool = "pen"
        
        self.setStyleSheet("border: 2px solid #ccc; background-color: white;")
        
    def set_tool(self, tool):
        """Встановлення інструменту (pen/eraser)"""
        self.current_tool = tool
        
    def set_pen_width(self, width):
        """Встановлення товщини пера"""
        self.pen_width = width
        
    def mousePressEvent(self, event):  # type: ignore[override]
        """Обробник натискання миші"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            
    def mouseMoveEvent(self, event):  # type: ignore[override]
        """Обробник руху миші"""
        if self.drawing and event.buttons() & Qt.MouseButton.LeftButton:
            painter = QPainter(self.image)
            
            if self.current_tool == "pen":
                painter.setPen(QPen(self.pen_color, self.pen_width, 
                                  Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                                  Qt.PenJoinStyle.RoundJoin))
            else:  # eraser
                painter.setPen(QPen(Qt.GlobalColor.white, self.pen_width * 2,
                                  Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                                  Qt.PenJoinStyle.RoundJoin))
            
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()
            
    def mouseReleaseEvent(self, event):  # type: ignore[override]
        """Обробник відпускання миші"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            
    def paintEvent(self, event):  # type: ignore[override]
        """Малювання віджету"""
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image, self.image.rect())
        
    def clear(self):
        """Очищення полотна"""
        self.image.fill(Qt.GlobalColor.white)
        self.update()
        
    def resizeEvent(self, event):  # type: ignore[override]
        """Обробник зміни розміру"""
        if self.size() != self.image.size():
            new_image = QImage(self.size(), QImage.Format.Format_RGB32)
            new_image.fill(Qt.GlobalColor.white)
            
            painter = QPainter(new_image)
            painter.drawImage(QPoint(0, 0), self.image)
            self.image = new_image
        
        super().resizeEvent(event)
        
    def save_to_temp(self) -> Optional[str]:
        """
        Збереження полотна у тимчасовий файл з автоматичним видаленням
        
        Returns:
            Шлях до тимчасового файлу або None
        """
        # Перевірка, чи є щось намальоване
        if self.is_empty():
            return None
            
        # Використовуємо NamedTemporaryFile для автоматичного видалення
        # Але зберігаємо файл відкритим до завершення обробки
        try:
            # Створюємо тимчасовий файл, який не буде автоматично видалений
            # (delete=False), щоб він був доступний для OCR
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.png',
                prefix='handwrite_canvas_',
                delete=False  # Не видаляти автоматично
            )
            temp_path = temp_file.name
            temp_file.close()
            
            # Зберігаємо зображення
            if self.image.save(temp_path):
                # Зберігаємо шлях для подальшого видалення
                if not hasattr(self, '_temp_files'):
                    self._temp_files = []
                self._temp_files.append(temp_path)
                return temp_path
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Помилка створення тимчасового файлу: {e}")
        
        return None
    
    def cleanup_temp_files(self):
        """Видалення всіх тимчасових файлів, створених цим екземпляром"""
        if hasattr(self, '_temp_files'):
            for temp_path in self._temp_files:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Не вдалося видалити тимчасовий файл {temp_path}: {e}")
            self._temp_files.clear()
        
    def is_empty(self):
        """Перевірка, чи порожнє полотно"""
        # Перевірка, чи є хоча б один небілий піксель
        # Використовуємо більш ефективну перевірку - перевіряємо кожен 5-й піксель
        white_color = QColor(Qt.GlobalColor.white)
        step = 5  # Зменшуємо крок для кращої перевірки
        
        for x in range(0, self.image.width(), step):
            for y in range(0, self.image.height(), step):
                pixel_color = self.image.pixelColor(x, y)
                # Перевіряємо, чи піксель не білий (з невеликою толерантністю для антиаліасингу)
                if pixel_color.rgb() != white_color.rgb():
                    # Додаткова перевірка: якщо піксель не білий, перевіряємо сусідні
                    # щоб уникнути помилок через антиаліасинг
                    if pixel_color.red() < 250 or pixel_color.green() < 250 or pixel_color.blue() < 250:
                        return False
        return True
