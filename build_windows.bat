@echo off
rem Builds the portable dist\Duckstoken.exe (engine bundled). Needs Python 3.10+ on PATH.
python -m pip install --upgrade pip -r requirements-build.txt || exit /b 1
python tools\fetch_engine.py || exit /b 1
python -m PyInstaller --noconfirm --clean --onefile --windowed --name Duckstoken --icon assets\duckstoken.ico ^
  --add-data "duckstoken/web;duckstoken/web" --add-data "engine;engine" ^
  --hidden-import pystray._win32 --collect-data webview duckstoken_launcher.py || exit /b 1
echo.
echo Done: dist\Duckstoken.exe
