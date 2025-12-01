# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec файл для Handwrite2Print
handwrite2print/build.spec

Використання:
    pyinstaller build.spec
"""

# PyInstaller injects these classes at runtime: Analysis, PYZ, EXE, COLLECT
# type: ignore

block_cipher = None

a = Analysis(  # type: ignore
    ['app/handwrite_main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('resources', 'resources'),
    ],
    hiddenimports=[
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui', 
        'PyQt6.QtWidgets',
        'cv2',
        'numpy',
        'PIL',
        'pytesseract',
        'easyocr',
        'paddleocr',
        'docx',
        'reportlab',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'IPython',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(  # type: ignore
    a.pure, 
    a.zipped_data,
    cipher=block_cipher
)

exe = EXE(  # type: ignore
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Handwrite2Print',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Приховати консоль
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='resources/icon.ico'  # Додайте іконку, якщо є
)

# Для створення окремої папки з залежностями замість одного файлу,
# використовуйте альтернативну конфігурацію PyInstaller
#     name='Handwrite2Print'
# )
