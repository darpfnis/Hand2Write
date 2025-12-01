"""
Початкове меню програми
handwrite2print/app/view/welcome_screen.py
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QMouseEvent


class WelcomeScreen(QWidget):
    """Початкове меню з вибором режиму роботи"""
    
    # Сигнали для вибору режиму
    draw_mode_selected = pyqtSignal()
    upload_mode_selected = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Ініціалізація інтерфейсу"""
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(30)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Верхня синя смуга (декоративна)
        top_bar = QFrame()
        top_bar.setFixedHeight(6)
        top_bar.setStyleSheet("background-color: #2196F3; border-radius: 3px;")
        layout.addWidget(top_bar)
        
        layout.addSpacing(20)
        
        # Заголовок
        title = QLabel("Система розпізнавання рукописного тексту")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont("Segoe UI", 32, QFont.Weight.Bold)
        title.setFont(title_font)
        title.setStyleSheet("color: #1976D2; background-color: transparent;")
        layout.addWidget(title)
        
        # Декоративна лінія під заголовком
        line = QFrame()
        line.setFixedSize(200, 2)
        line.setStyleSheet("background-color: #90CAF9;")
        line_container = QWidget()
        line_layout = QHBoxLayout(line_container)
        line_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        line_layout.addWidget(line)
        layout.addWidget(line_container)
        
        layout.addSpacing(10)
        
        # Підзаголовок
        subtitle = QLabel("Оберіть режим роботи")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_font = QFont("Segoe UI", 16)
        subtitle.setFont(subtitle_font)
        subtitle.setStyleSheet("color: #546E7A; background-color: transparent;")
        layout.addWidget(subtitle)
        
        layout.addSpacing(30)
        
        # Контейнер для кнопок
        buttons_container = QWidget()
        buttons_layout = QHBoxLayout(buttons_container)
        buttons_layout.setSpacing(40)
        buttons_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Кнопка "Малювати"
        draw_button = self.create_mode_button(
            "Малювати",
            "Створіть рукописний текст\nна цифровому полотні",
            "#2196F3",
            self.draw_mode_selected.emit
        )
        buttons_layout.addWidget(draw_button)
        
        # Кнопка "Завантажити зображення"
        upload_button = self.create_mode_button(
            "Завантажити зображення",
            "Завантажте готове зображення\nз рукописним текстом",
            "#4CAF50",
            self.upload_mode_selected.emit
        )
        buttons_layout.addWidget(upload_button)
        
        layout.addWidget(buttons_container)
        layout.addStretch()
        
        # Стилізація фону (біло-блакитний градієнт)
        self.setStyleSheet("""
            WelcomeScreen {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8f9fa,
                    stop:1 #e8f4f8
                );
            }
        """)
        
    def create_mode_button(self, title, description, color, callback):
        """Створення кнопки режиму"""
        button_frame = ClickableFrame(title, callback)
        button_frame.setFixedSize(340, 200)
        button_frame.setStyleSheet(f"""
            ClickableFrame {{
                background-color: white;
                border: 3px solid {color};
                border-radius: 15px;
            }}
            ClickableFrame:hover {{
                background-color: #f5f5f5;
                border: 3px solid {color};
            }}
        """)
        
        frame_layout = QVBoxLayout(button_frame)
        frame_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frame_layout.setSpacing(15)
        frame_layout.setContentsMargins(20, 30, 20, 30)
        
        # Заголовок кнопки
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont("Segoe UI", 18, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet(f"color: {color}; background-color: transparent;")
        title_label.setWordWrap(True)
        frame_layout.addWidget(title_label)
        
        # Опис
        desc_label = QLabel(description)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setWordWrap(True)
        desc_label.setMinimumHeight(50)
        desc_font = QFont("Segoe UI", 11)
        desc_label.setFont(desc_font)
        desc_label.setStyleSheet(f"color: #546E7A; background-color: transparent; padding: 5px;")
        frame_layout.addWidget(desc_label)
        
        return button_frame


class ClickableFrame(QFrame):
    """Клікабельний фрейм для кнопок режиму"""
    
    def __init__(self, title, callback, parent=None):
        super().__init__(parent)
        self.title = title
        self.callback = callback
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
    def mousePressEvent(self, a0):
        """Обробник натискання миші"""
        if a0 and a0.button() == Qt.MouseButton.LeftButton:
            self.callback()
        super().mousePressEvent(a0)