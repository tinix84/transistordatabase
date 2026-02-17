# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the TDB PyQt5 desktop GUI."""

import glob
import os

block_cipher = None

# Collect GUI resource files
gui_dir = os.path.join('..', 'transistordatabase', 'gui')
gui_datas = []
for pattern in ('*.ui', '*.png', 'settings.json'):
    for f in glob.glob(os.path.join(gui_dir, pattern)):
        gui_datas.append((f, 'transistordatabase/gui'))

a = Analysis(
    ['../transistordatabase/scripts/tdb_pyqt.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('../transistordatabase/database', 'transistordatabase/database'),
        *gui_datas,
    ],
    hiddenimports=[
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'PyQt5.QtWebEngineWidgets',
        'PyQt5.sip',
        'transistordatabase',
        'transistordatabase.core',
        'transistordatabase.core.models',
        'transistordatabase.core.repository',
        'transistordatabase.core.adapters',
        'transistordatabase.core.services',
        'transistordatabase.backend',
        'transistordatabase.backend.concrete_services',
        'transistordatabase.gui.gui',
        'transistordatabase.gui.api_client',
        'transistordatabase.gui._widgets',
        'transistordatabase.gui._utils',
        'transistordatabase.gui.mixins.utils_mixin',
        'transistordatabase.gui.mixins.settings_mixin',
        'transistordatabase.gui.mixins.search_mixin',
        'transistordatabase.gui.mixins.creation_mixin',
        'transistordatabase.gui.mixins.curve_mixin',
        'transistordatabase.gui.mixins.export_mixin',
        'transistordatabase.gui.mixins.comparison_mixin',
        'transistordatabase.gui.mixins.topology_mixin',
        'matplotlib',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.figure',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'uvicorn',
        'fastapi',
        'tkinter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='tdb-pyqt',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
