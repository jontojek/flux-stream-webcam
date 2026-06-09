@echo off
title flux-stream-webcam
color 0A

set PYTHON=C:\Users\jonto\miniconda3\python.exe
set HF_HOME=D:\AI_software\Github_repos\flux-stream-webcam\models\hf_cache
set HUGGINGFACE_HUB_CACHE=D:\AI_software\Github_repos\flux-stream-webcam\models\hf_cache\hub

echo.
echo  ============================================================
echo   flux-stream-webcam  --  Starting up
echo  ============================================================
echo.

echo  [1/3] Checking Python packages...
%PYTHON% -c "import torch" 2>nul
if errorlevel 1 (
    echo  !! Packages not installed. Double-click INSTALL.bat first.
    pause & exit /b 1
)
echo        OK
echo.

echo  [2/3] Checking GPU...
%PYTHON% -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU found'"
if errorlevel 1 (
    echo  !! No CUDA GPU detected. Check your drivers.
    pause & exit /b 1
)
echo        OK
echo.

echo  [3/3] Launching...
echo.
echo  --------------------------------------------------------
echo   Model: FLUX.2-Klein-4B (~15 GB, already in models\hf_cache\).
echo   First launch compiles GPU kernels (1-3 min); cached after.
echo.
echo   ~5 FPS at 2 steps / ~8 FPS at 1 step (512px) on the 5090.
echo   Speak to set the style.  Live controls:
echo     +/- steps    [ ] strength    , . feedback    Q quit
echo   strength = webcam vs restyle;  feedback = trails / morphing.
echo  --------------------------------------------------------
echo.

%PYTHON% main.py

echo.
echo  Exited.
pause
