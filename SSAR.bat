@echo off
REM Активировать виртуальное окружение
call .venv\Scripts\activate

REM Установить зависимости
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

REM Запустить основной скрипт
python main.py

pause
