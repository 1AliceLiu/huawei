FROM ultralytics/ultralytics@sha256:3e4ac86e8e3a451f9fad2e6cb4ecb164600b6de4a558eb3b1d828f68af6dc557

RUN python -m pip install --no-cache-dir --no-deps --force-reinstall \
    ultralytics==8.4.130 av==19.0.1 lap==0.5.13

WORKDIR /app
COPY realtime_web.py region_calibration.py traffic_api.py traffic_count.py traffic_push.py record_traffic.py stage_timing.py video_recorder.py ./
COPY web/ ./web/
COPY tests/ ./tests/
COPY scripts/receive_traffic_counts.py ./scripts/receive_traffic_counts.py

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    YOLO_CONFIG_DIR=/tmp \
    YOLO_AUTOINSTALL=false \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

USER 10001:10001
ENTRYPOINT ["python", "-X", "utf8", "realtime_web.py"]
