"""
Вікно завантаження (Splash Screen)
handwrite2print/app/view/splash_screen.py
"""
import sys
from PyQt6.QtWidgets import QSplashScreen, QApplication
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QPainter, QFont, QColor, QLinearGradient
from pathlib import Path

# Константа для шрифту
DEFAULT_FONT = "Segoe UI"


class SplashScreen(QSplashScreen):
    """Вікно завантаження програми"""
    
    def __init__(self):
        # Створюємо pixmap для splash screen
        pixmap = QPixmap(800, 500)
        pixmap.fill(QColor("#ffffff"))
        
        # Малюємо контент на pixmap
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        
        # Градієнтний фон
        gradient = QLinearGradient(0, 0, 0, 500)
        gradient.setColorAt(0, QColor("#f8f9fa"))
        gradient.setColorAt(1, QColor("#e8f4f8"))
        painter.fillRect(pixmap.rect(), gradient)
        
        # Синя декоративна смуга зверху
        painter.fillRect(0, 0, 800, 8, QColor("#2196F3"))
        
        # Рамка
        border_pen = painter.pen()
        border_pen.setWidth(2)
        border_pen.setColor(QColor("#2196F3"))
        painter.setPen(border_pen)
        painter.drawRoundedRect(15, 15, 770, 470, 10, 10)
        
        # Заголовок - розбиваємо на дві частини для кращого контролю
        title_font = QFont(DEFAULT_FONT, 32, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor("#1976D2"))
        
        # Перша частина заголовку
        painter.drawText(
            50, 120, 700, 50,
            Qt.AlignmentFlag.AlignCenter,
            "Система розпізнавання"
        )
        
        # Друга частина заголовку
        painter.drawText(
            50, 165, 700, 50,
            Qt.AlignmentFlag.AlignCenter,
            "рукописного тексту"
        )
        
        # Декоративна лінія
        painter.setPen(QColor("#90CAF9"))
        painter.drawLine(250, 235, 550, 235)
        
        # Підзаголовок
        subtitle_font = QFont(DEFAULT_FONT, 14)
        painter.setFont(subtitle_font)
        painter.setPen(QColor("#546E7A"))
        painter.drawText(
            50, 270, 700, 30,
            Qt.AlignmentFlag.AlignCenter,
            "Завантаження програми..."
        )
        
        # Індикатор завантаження
        loading_font = QFont(DEFAULT_FONT, 12)
        painter.setFont(loading_font)
        painter.setPen(QColor("#42A5F5"))
        painter.drawText(
            50, 340, 700, 30,
            Qt.AlignmentFlag.AlignCenter,
            "Ініціалізація компонентів..."
        )
        
        # Повідомлення внизу
        progress_font = QFont(DEFAULT_FONT, 10)
        painter.setFont(progress_font)
        painter.setPen(QColor("#90A4AE"))
        painter.drawText(
            50, 420, 700, 30,
            Qt.AlignmentFlag.AlignCenter,
            "Будь ласка, зачекайте..."
        )
        
        painter.end()
        
        super().__init__(pixmap)
        
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.SplashScreen |
            Qt.WindowType.FramelessWindowHint
        )
        
        # Центруємо splash screen на екрані
        try:
            screen = QApplication.primaryScreen()
            if screen:
                screen_geometry = screen.geometry()
                x = (screen_geometry.width() - pixmap.width()) // 2
                y = (screen_geometry.height() - pixmap.height()) // 2
                self.move(x, y)
        except Exception:
            pass
        
        # Зберігаємо початковий pixmap для оновлення
        self._base_pixmap = pixmap.copy()
        
        # Показуємо splash screen
        self.show()
        QApplication.processEvents()
    
    def update_message(self, message: str):
        """Оновлення повідомлення на splash screen"""
        # Створюємо копію базового pixmap
        pixmap = QPixmap(self._base_pixmap)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        
        # Очищаємо область повідомлення
        gradient = QLinearGradient(0, 330, 0, 370)
        gradient.setColorAt(0, QColor("#f8f9fa"))
        gradient.setColorAt(1, QColor("#e8f4f8"))
        painter.fillRect(50, 330, 700, 40, gradient)
        
        # Малюємо нове повідомлення
        loading_font = QFont(DEFAULT_FONT, 12)
        painter.setFont(loading_font)
        painter.setPen(QColor("#42A5F5"))
        painter.drawText(
            50, 340, 700, 30,
            Qt.AlignmentFlag.AlignCenter,
            message
        )
        
        painter.end()
        self.setPixmap(pixmap)
        QApplication.processEvents()
    
    def close_splash(self):
        """Закриття splash screen"""
        self.finish(None)


if __name__ == "__main__":
    # Цей файл не призначений для прямого запуску
    print("Цей файл не призначений для прямого запуску.")
    print("Запустіть програму через: python app/handwrite_main.py")
    sys.exit(1)
