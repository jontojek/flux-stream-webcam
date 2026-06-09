@echo off
title flux-stream-webcam  INSTALLER
color 0A

:: Use miniconda Python directly to avoid picking up the wrong install
set PYTHON=C:\Users\jonto\miniconda3\python.exe
set PIP=C:\Users\jonto\miniconda3\python.exe -m pip

:: Keep HuggingFace cache inside the project folder
set HF_HOME=D:\AI_software\Github_repos\flux-stream-webcam\models\hf_cache
set HUGGINGFACE_HUB_CACHE=D:\AI_software\Github_repos\flux-stream-webcam\models\hf_cache\hub

echo.
echo  ============================================================
echo   flux-stream-webcam  --  First-Time Setup
echo  ============================================================
echo.
echo  Using Python at: %PYTHON%
echo.
echo  This will install all the Python packages needed to run.
echo  It only needs to be done ONCE.
echo.
echo  Step 1 of 2: Installing PyTorch (the AI engine)
echo  This is a large download (~2.5 GB) -- please be patient.
echo.

%PIP% install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

if errorlevel 1 (
    echo.
    echo  !! ERROR: PyTorch failed to install.
    echo     Make sure you have an internet connection.
    echo.
    pause
    exit /b 1
)

echo.
echo  Step 2 of 2: Installing remaining packages...
echo.

%PIP% install "torchao>=0.9.0" "diffusers>=0.31.0" "transformers>=4.46.0" accelerate peft sounddevice faster-whisper opencv-python Pillow numpy triton

if errorlevel 1 (
    echo.
    echo  !! ERROR: Some packages failed to install.
    echo     Check the messages above for details.
    echo.
    pause
    exit /b 1
)

echo.
echo  ============================================================
echo   All done!  You can now double-click CHECK.bat to verify,
echo   then START.bat to run.
echo  ============================================================
echo.
pause
