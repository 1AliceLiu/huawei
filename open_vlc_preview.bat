@echo off
rem 直接用本机 VLC 打开预览流（需先启动 run_preview.bat）
start "" "D:\VLC\vlc.exe" --network-caching=1000 "rtsp://127.0.0.1:8554/preview"
