"""
Панель інструментів для головного вікна
handwrite2print/app/view/handwrite_toolbar.py
"""
from PyQt6.QtWidgets import (QToolBar, QPushButton, QWidget, QSizePolicy)
from PyQt6.QtCore import Qt


class HandwriteToolBar(QToolBar):
    """Панель інструментів для програми розпізнавання рукописного тексту"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMovable(False)
        self.init_ui()
        
    def init_ui(self):
        """Ініціалізація інтерфейсу toolbar"""
        # Кнопка повернення до меню (вліво)
        self.back_btn = QPushButton("Назад до меню")
        self.back_btn.setToolTip("Повернутися до початкового меню")
        self.addWidget(self.back_btn)
        
        # Separator після кнопки "Назад до меню"
        self.addSeparator()
        
        # Лівий spacer для центрування
        left_spacer = QWidget()
        left_spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.addWidget(left_spacer)
        
        # Кнопка завантаження (буде додана/видалена динамічно вправо)
        self.load_btn = QPushButton("Завантажити")
        self.load_btn.setToolTip("Завантажити зображення з файлу")
        self.load_separator_before = None
        self.load_separator_after = None
        self.load_btn_added = False  # Прапорець, чи додана кнопка до toolbar
        
        # Кнопка розпізнавання
        self.convert_btn = QPushButton("Розпізнати текст")
        self.convert_btn.setToolTip("Запустити розпізнавання тексту")
        self.convert_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 8px 20px;")
        self.addWidget(self.convert_btn)
        
        self.addSeparator()
        
        # Кнопка збереження
        self.export_btn = QPushButton("Зберегти")
        self.export_btn.setToolTip("Зберегти результат у файл")
        self.addWidget(self.export_btn)
        
        self.addSeparator()
        
        # Кнопка очищення
        self.clear_btn = QPushButton("Очистити")
        self.clear_btn.setToolTip("Очистити всі дані")
        self.addWidget(self.clear_btn)
        
        # Правий spacer для центрування
        right_spacer = QWidget()
        right_spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.right_spacer = right_spacer  # Зберігаємо посилання для додавання кнопки завантаження
        self.addWidget(right_spacer)
    
    def show_for_draw_mode(self):
        """Показати toolbar для режиму малювання (без кнопки завантаження)"""
        self.setVisible(True)
        # Завжди перевіряємо та видаляємо кнопку завантаження, навіть якщо load_btn_added = False
        # (на випадок, якщо кнопка була додана в іншому місці)
        try:
            # Знаходимо та видаляємо action для кнопки завантаження
            # Використовуємо widgetForAction() для отримання віджету з action
            load_action = None
            for action in self.actions():
                try:
                    # Використовуємо widgetForAction() для QToolBar
                    widget = self.widgetForAction(action)
                    if widget == self.load_btn:
                        load_action = action
                        break
                except (AttributeError, RuntimeError, TypeError):
                    continue
            
            # Видаляємо кнопку завантаження, якщо знайдена
            if load_action:
                # Знаходимо індекси separators навколо кнопки
                actions_list = list(self.actions())
                try:
                    load_idx = actions_list.index(load_action)
                    # Видаляємо separator перед кнопкою (якщо є)
                    if load_idx > 0:
                        prev_action = actions_list[load_idx - 1]
                        if prev_action.isSeparator():
                            self.removeAction(prev_action)
                    # Видаляємо кнопку завантаження
                    self.removeAction(load_action)
                    # Видаляємо separator після кнопки (якщо є)
                    if load_idx < len(actions_list) - 1:
                        next_action = actions_list[load_idx + 1]
                        if next_action.isSeparator():
                            self.removeAction(next_action)
                except (ValueError, IndexError, AttributeError):
                    # Якщо не вдалося знайти індекси, просто видаляємо кнопку
                    self.removeAction(load_action)
            
            # Також видаляємо збережені посилання на separators
            if self.load_separator_before:
                try:
                    self.removeAction(self.load_separator_before)
                except Exception:
                    pass
            if self.load_separator_after:
                try:
                    self.removeAction(self.load_separator_after)
                except Exception:
                    pass
            
            self.load_separator_before = None
            self.load_separator_after = None
            self.load_btn_added = False
            
            # Додаткова перевірка - переконаємося, що кнопка дійсно видалена
            for action in self.actions():
                try:
                    # Використовуємо widgetForAction() для QToolBar
                    widget = self.widgetForAction(action)
                    if widget == self.load_btn:
                        # Якщо кнопка все ще знайдена, видаляємо її
                        self.removeAction(action)
                except (AttributeError, RuntimeError, TypeError):
                    pass
        except (RuntimeError, AttributeError):
            pass
    
    def show_for_upload_mode(self):
        """Показати toolbar для режиму завантаження (з кнопкою завантаження)"""
        self.setVisible(True)
        # Додаємо кнопку завантаження та separators, якщо вони не додані
        if not self.load_btn_added:
            try:
                # Знаходимо action для кнопки "Очистити" (щоб додати кнопку завантаження після неї)
                clear_action = None
                right_spacer_action = None
                for action in self.actions():
                    try:
                        # Використовуємо widgetForAction() для QToolBar
                        widget = self.widgetForAction(action)
                        if widget == self.clear_btn:
                            clear_action = action
                        elif widget == self.right_spacer:
                            right_spacer_action = action
                    except (AttributeError, RuntimeError, TypeError):
                        continue
                
                # Якщо знайдено right_spacer, додаємо перед ним
                if right_spacer_action:
                    # Додаємо separator перед кнопкою завантаження
                    self.load_separator_before = self.insertSeparator(right_spacer_action)
                    # Додаємо кнопку завантаження
                    load_action = self.insertWidget(self.load_separator_before, self.load_btn)
                    # Додаємо separator після кнопки завантаження
                    self.load_separator_after = self.insertSeparator(load_action)
                    self.load_btn_added = True
                # Якщо знайдено clear_action, додаємо після нього (перед right_spacer)
                elif clear_action:
                    # Знаходимо наступний action після clear_action (це має бути right_spacer)
                    actions_list = list(self.actions())
                    try:
                        clear_idx = actions_list.index(clear_action)
                        if clear_idx < len(actions_list) - 1:
                            next_action = actions_list[clear_idx + 1]
                            # Додаємо separator перед наступним action
                            self.load_separator_before = self.insertSeparator(next_action)
                            # Додаємо кнопку завантаження
                            load_action = self.insertWidget(self.load_separator_before, self.load_btn)
                            # Додаємо separator після кнопки завантаження
                            self.load_separator_after = self.insertSeparator(load_action)
                            self.load_btn_added = True
                        else:
                            # Якщо clear_action останній, додаємо після нього
                            self.load_separator_before = self.insertSeparator(clear_action)
                            load_action = self.insertWidget(self.load_separator_before, self.load_btn)
                            self.load_separator_after = self.insertSeparator(load_action)
                            self.load_btn_added = True
                    except (ValueError, IndexError):
                        # Якщо не вдалося знайти індекс, додаємо після clear_action
                        self.load_separator_before = self.insertSeparator(clear_action)
                        load_action = self.insertWidget(self.load_separator_before, self.load_btn)
                        self.load_separator_after = self.insertSeparator(load_action)
                        self.load_btn_added = True
                else:
                    # Якщо нічого не знайдено, додаємо в кінець
                    self.load_separator_before = self.addSeparator()
                    self.addWidget(self.load_btn)
                    self.load_separator_after = self.addSeparator()
                    self.load_btn_added = True
            except (RuntimeError, AttributeError):
                # Якщо не вдалося вставити, просто додаємо в кінець
                try:
                    self.load_separator_before = self.addSeparator()
                    self.addWidget(self.load_btn)
                    self.load_separator_after = self.addSeparator()
                    self.load_btn_added = True
                except Exception:
                    pass
    
    def hide_toolbar(self):
        """Приховати toolbar (для головного меню)"""
        self.setVisible(False)
