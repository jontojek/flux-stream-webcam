@echo off
title flux-stream-webcam  MIC CHECK
color 0B

:: Use miniconda Python directly (bare 'python' picks up the wrong install)
set PYTHON=C:\Users\jonto\miniconda3\python.exe

echo.
echo  ============================================================
echo   flux-stream-webcam  --  Microphone Level Meter
echo  ============================================================
echo.
echo   Talk normally. The bar should jump past the ^| marker.
echo   Note your level, then set AUDIO_SILENCE_THRESHOLD in config.py.
echo.
echo   Optional: pass a device index to test a specific mic, e.g.
echo     MIC_CHECK.bat 3
echo.

%PYTHON% mic_check.py %1

pause
