# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Analyse Overselection application.

Builds 3 executables:
  - interface.exe  (main GUI - entry point for users)
  - main_algo.exe  (analysis engine, called by interface)
  - roi_selector.exe (ROI configuration, called by main_algo)

Usage:
  pyinstaller build.spec
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files

# Increase recursion limit for large dependency trees
sys.setrecursionlimit(10000)

block_cipher = None

# Collect ultralytics data files (YOLO model configs, etc.)
ultralytics_datas = collect_data_files('ultralytics')

# Common data files bundled with each executable
common_datas = [
    ('renault_group.png', '.'),
    ('yolov8x-seg.pt', '.'),
]

# Per-target hidden imports — only declare what each script actually uses.
# This avoids PyInstaller re-analyzing torch/ultralytics 3 times.

# interface.py: just a tkinter GUI that launches main_algo via subprocess
interface_hiddenimports = [
    'tkinter', 'frozen_utils',
]

# main_algo.py: the heavy analysis engine (needs ML + reporting)
main_algo_hiddenimports = [
    'cv2', 'numpy', 'pandas', 'openpyxl', 'PIL',
    'matplotlib', 'matplotlib.backends.backend_agg',
    'reportlab', 'reportlab.lib.pagesizes', 'reportlab.pdfgen',
    'asammdf', 'mat73', 'gzip', 'shutil',
    'tkinter', 'json', 're',
    'ultralytics',
    'torch', 'torchvision',
    'frozen_utils','builtins'
]

# roi_selector.py: tkinter GUI for ROI selection (no ML needed)
roi_selector_hiddenimports = [
    'tkinter', 'cv2', 'numpy', 'PIL', 'pandas', 'json', 're',
    'openpyxl', 'openpyxl.styles', 'openpyxl.utils', 'et_xmlfile',
]

# threshold_tuner.py: tkinter GUI for threshold calibration (no ML needed)
threshold_tuner_hiddenimports = [
    'tkinter', 'cv2', 'numpy', 'PIL', 'json', 're',
    'masks', 'frozen_utils',
]

# Exclude packages your project does NOT need — critical for anaconda envs
common_excludes = [
    'tensorflow', 'tensorboard', 'keras',
    'transformers', 'torchaudio',
    'boto3', 'botocore', 's3transfer',
    'sklearn', 'scikit-learn', 'gevent',
    'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
    'IPython', 'jupyter', 'notebook', 'ipykernel',
    'sphinx', 'docutils',
    'altair', 'narwhals',
    'pydantic',
    'sympy',
    'pytest',
]

# Extra excludes for lightweight targets (they don't need ML/reporting)
interface_excludes = common_excludes + [
    'torch', 'torchvision', 'ultralytics',
    'cv2', 'numpy', 'pandas', 'openpyxl', 'PIL',
    'matplotlib', 'reportlab', 'asammdf', 'mat73', 'scipy',
]

roi_selector_excludes = common_excludes + [
    'torch', 'torchvision', 'ultralytics',
    'matplotlib', 'reportlab', 'asammdf', 'mat73',
    'scipy',
]

threshold_tuner_excludes = common_excludes + [
    'torch', 'torchvision', 'ultralytics',
    'matplotlib', 'reportlab', 'asammdf', 'mat73',
    'openpyxl', 'scipy', 'pandas',
]

# ========================================
# 1. interface.exe (main GUI)
# ========================================
interface_a = Analysis(
    ['scripts/interface.py'],
    pathex=['.'],
    binaries=[],
    datas=[('renault_group.png', '.')],
    hiddenimports=interface_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=interface_excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ========================================
# 2. main_algo.exe (analysis engine)
# ========================================
main_algo_a = Analysis(
    ['scripts/main_algo.py'],
    pathex=['.'],
    binaries=[],
    datas=common_datas + ultralytics_datas,
    hiddenimports=main_algo_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=common_excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ========================================
# 3. roi_selector.exe (ROI configuration)
# ========================================
roi_selector_a = Analysis(
    ['scripts/roi_selector.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=roi_selector_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=roi_selector_excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ========================================
# 4. threshold_tuner.exe (threshold calibration)
# ========================================
threshold_tuner_a = Analysis(
    ['scripts/threshold_tuner.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=threshold_tuner_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=threshold_tuner_excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ========================================
# MERGE all analyses to share common files (reduces total size)
# ========================================
MERGE(
    (interface_a, 'interface', 'interface'),
    (main_algo_a, 'main_algo', 'main_algo'),
    (roi_selector_a, 'roi_selector', 'roi_selector'),
    (threshold_tuner_a, 'threshold_tuner', 'threshold_tuner'),
)

# --- interface ---
interface_pyz = PYZ(interface_a.pure, interface_a.zipped_data, cipher=block_cipher)
interface_exe = EXE(
    interface_pyz,
    interface_a.scripts,
    [],
    exclude_binaries=True,
    name='interface',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window for GUI app
    icon=None,      # Add your .ico file here if you have one: icon='app.ico'
)

# --- main_algo ---
main_algo_pyz = PYZ(main_algo_a.pure, main_algo_a.zipped_data, cipher=block_cipher)
main_algo_exe = EXE(
    main_algo_pyz,
    main_algo_a.scripts,
    [],
    exclude_binaries=True,
    name='main_algo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Keep console for subprocess stdout capture
)

# --- roi_selector ---
roi_selector_pyz = PYZ(roi_selector_a.pure, roi_selector_a.zipped_data, cipher=block_cipher)
roi_selector_exe = EXE(
    roi_selector_pyz,
    roi_selector_a.scripts,
    [],
    exclude_binaries=True,
    name='roi_selector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

# --- threshold_tuner ---
threshold_tuner_pyz = PYZ(threshold_tuner_a.pure, threshold_tuner_a.zipped_data, cipher=block_cipher)
threshold_tuner_exe = EXE(
    threshold_tuner_pyz,
    threshold_tuner_a.scripts,
    [],
    exclude_binaries=True,
    name='threshold_tuner',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

# ========================================
# Collect into a single output directory
# ========================================
coll = COLLECT(
    interface_exe, interface_a.binaries, interface_a.zipfiles, interface_a.datas,
    main_algo_exe, main_algo_a.binaries, main_algo_a.zipfiles, main_algo_a.datas,
    roi_selector_exe, roi_selector_a.binaries, roi_selector_a.zipfiles, roi_selector_a.datas,
    threshold_tuner_exe, threshold_tuner_a.binaries, threshold_tuner_a.zipfiles, threshold_tuner_a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AnalyseOverselection',
)
