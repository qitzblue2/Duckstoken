@echo off
rem Builds dist\Duckstoken.exe (needs Python 3.10+ on PATH).
python -m pip install --upgrade pip pyinstaller || exit /b 1
python -m PyInstaller --noconfirm --clean --onefile --console --name Duckstoken ^
  --add-data "duckstoken/web;duckstoken/web" duckstoken_launcher.py || exit /b 1
echo.
echo Done: dist\Duckstoken.exe
