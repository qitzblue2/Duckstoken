@echo off
rem Builds the portable dist\Duckstoken.exe (engine bundled). Needs Python 3.10+ on PATH.
python -m pip install --upgrade pip -r requirements-build.txt || exit /b 1
python tools\fetch_engine.py || exit /b 1
python tools\build_exe.py onefile || exit /b 1
echo.
echo Done: dist\Duckstoken.exe
echo Check it:  dist\Duckstoken.exe --selftest selftest.json  (then open selftest.json)
