@echo off
if not exist "venv" (
    echo Virtual environment not found. Please set it up first.
    exit /b 1
)
call venv\Scripts\activate
python -m stt.system_audio %*
pause
