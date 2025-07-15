@echo off
setlocal
cd /d "%~dp0"

python main.py --section ghost
python main.py --section talk
python main.py --section lego
python main.py --section animals
pause
