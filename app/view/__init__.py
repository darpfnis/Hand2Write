"""
Пакет інтерфейсу користувача
"""
from .handwrite_main_window import MainWindow
from .handwrite_canvas import CanvasWidget
from .handwrite_dialogs import SettingsDialog, AboutDialog

__all__ = [
    'MainWindow',
    'CanvasWidget',
    'SettingsDialog',
    'AboutDialog'
]

