# API

## Текущий модуль

Код API находится в `apps/api/api.py`. Для обратной совместимости в корне сохранён thin entrypoint `api.py`, поэтому команда `uvicorn api:app` продолжает работать.

## Endpoint'ы

- `GET /`
- `GET /health`
- `POST /predict`
- `POST /predict/batch`

## Поведение

API сохраняет существующую MVP-логику:

- загружает модель из `MODEL_PATH` или из `ml/artifacts/best_model.joblib` по умолчанию
- возвращает `expected_price_proxy`
- опционально считает `listing_price`, `delta_abs`, `delta_pct`
- поддерживает `USD`, `RUB`, `BOTH`
- не меняет контракт health/predict/batch

## Запуск

```bash
uvicorn apps.api.api:app --host 0.0.0.0 --port 8000
```

Совместимый вариант:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Docker

Канонический Dockerfile API находится в `apps/api/Dockerfile`. Он собирается из корня репозитория, чтобы в образ попадали и `apps/`, и `ml/`.

## Подробный контракт

Исторический файл [`API_REFERENCE.md`](../../API_REFERENCE.md) сохранён как точка входа и ссылается на эту документацию.
