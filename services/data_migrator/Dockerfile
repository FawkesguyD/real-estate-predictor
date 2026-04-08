FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app \
    CSV_SOURCE_PATH=./office-sale.csv

WORKDIR ${APP_HOME}

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY alembic.ini ./alembic.ini
COPY alembic ./alembic
COPY shared ./shared
COPY services ./services
COPY office-sale.csv ./office-sale.csv

CMD ["python", "-m", "services.data_migrator"]
