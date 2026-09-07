ARG PYTHON_BASE_IMAGE=docker.m.daocloud.io/python:3.12-slim-bookworm
FROM ${PYTHON_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=xcb \
    DISPLAY=:99 \
    ARTINSIGHT_CONFIG=/opt/artinsight/config/settings.aliyun.json

WORKDIR /opt/artinsight

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fluxbox \
        fonts-noto-cjk \
        libdbus-1-3 \
        libegl1 \
        libfontconfig1 \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libx11-6 \
        libx11-xcb1 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-xfixes0 \
        libxcb-xinerama0 \
        libxext6 \
        libxkbcommon-x11-0 \
        libxrender1 \
        novnc \
        procps \
        websockify \
        x11vnc \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml main.py ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY config ./config
COPY controllers ./controllers
COPY core ./core
COPY database ./database
COPY models ./models
COPY repositories ./repositories
COPY resources ./resources
COPY services ./services
COPY views ./views
COPY visualization ./visualization
COPY deploy/novnc/start-artinsight.sh /usr/local/bin/start-artinsight

RUN chmod +x /usr/local/bin/start-artinsight \
    && mkdir -p data/cache data/models data/exports logs

EXPOSE 6080

CMD ["start-artinsight"]
