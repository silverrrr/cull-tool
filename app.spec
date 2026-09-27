# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置: 选图工具 (单 exe, 不依赖 venv)。"""

import os
import sys

from PyInstaller.utils.hooks import collect_all

TOOL = os.path.dirname(os.path.abspath(SPECPATH)) if 'SPECPATH' in dir() else \
       os.path.dirname(os.path.abspath(__file__))
ENTRY = os.path.join(TOOL, 'app.py')

block_cipher = None

# webview 的 js/ 和 lib/ (WebView2 的 .NET 封装 dll + WebView2Loader) 都是运行时
# 按 realpath(__file__) 去读的, 放进 base_library.zip 就读不到 -> 界面直接白屏。
# pythonnet 同理 (它的 .NET 运行时 dll 必须落在磁盘上)。
wv_datas, wv_bins, wv_hidden = collect_all('webview')
pn_datas, pn_bins, pn_hidden = collect_all('pythonnet')

datas = wv_datas + pn_datas + [(os.path.join(TOOL, 'ui.html'), '.')]
binaries = wv_bins + pn_bins
hiddenimports = sorted(set(
    ['rawpy', 'PIL._tkinter_finder', 'PIL.Image',
     'clr', 'clr_loader',
     'webview.platforms.winforms', 'webview.platforms.edgechromium']
    + wv_hidden + pn_hidden))

a = Analysis(
    [ENTRY],
    pathex=[TOOL],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # 这些是拖进 venv 的打包工具用不到的, 排掉能小很多
    excludes=[
        'matplotlib', 'scipy', 'pandas', 'tkinter.test', 'test', 'unittest',
        'PIL.ImageQt', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
        'IPython', 'notebook', 'pydoc_data',
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
    name='选图工具',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,        # 图形程序, 不带控制台
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
