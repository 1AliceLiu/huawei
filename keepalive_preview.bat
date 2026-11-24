@echo off
rem Keepalive wrapper: restart realtime_web.py 5s after it exits/crashes.
cd /d "%~dp0"
:loop
.venv\Scripts\python.exe -X utf8 realtime_web.py --model runs/detect/yolo26m_merged12_best_20261021_0957/weights/best.pt --sources-file huawei-rtsp.txt --imgsz 960 --half --device 0 --port 8000
timeout /t 5 /nobreak >nul
goto loop
