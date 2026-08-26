# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:\\Users\\2bpro\\Desktop\\Release_v2_altitude\\Release_v2\\submission\\highdream_launcher.py'],
    pathex=['C:\\Users\\2bpro\\Desktop\\Release_v2_altitude\\Release_v2', 'C:\\Users\\2bpro\\Desktop\\Release_v2_altitude\\Release_v2\\src'],
    binaries=[],
    datas=[('C:\\Users\\2bpro\\Desktop\\Release_v2_altitude\\Release_v2\\artifacts\\models\\highdream\\altitude_attack_followup_v1_stage6obs19_v8_angle090_opp055_50iter_C10', 'artifacts\\models\\highdream\\altitude_attack_followup_v1_stage6obs19_v8_angle090_opp055_50iter_C10')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='HighDream_FINAL',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
