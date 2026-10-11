@echo off
rem Run Duckstoken straight from source (no build needed). Requires Python 3.10+.
title Duckstoken
where py >nul 2>nul && (py -3 -m duckstoken %* & goto :eof)
python -m duckstoken %*
