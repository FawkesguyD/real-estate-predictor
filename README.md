# Платформа предсказания цены на объект недвижимости

Сейчас в репозитории уже выделены отдельный API-модуль и ML-часть; позже рядом можно добавлять `apps/ui` и другие сервисы без смены базовой структуры.

## Структура репозитория

```text
.
├── apps/
│   └── api/                 # FastAPI-модуль и его Dockerfile
├── docs/
│   ├── api/
│   ├── architecture/
│   ├── deployment/
│   └── ml/
├── ml/
│   ├── artifacts/           # сохранённые model bundles для inference
│   └── model/               # training/inference код
├── .github/workflows/
├── api.py                   # совместимый entrypoint для uvicorn api:app
├── main.py                  # совместимый entrypoint для python main.py
├── docker-compose.yml
└── requirements.txt
```

## Какие модули уже есть

- `apps/api` — текущий FastAPI сервис с ручками `/`, `/health`, `/predict`, `/predict/batch`.
- `ml/model` — код обучения, inference, explainability и подготовки данных.
- `ml/artifacts` — бинарные артефакты модели, используемые API по умолчанию.

## Где лежат артефакты модели

Артефакты вынесены в `ml/artifacts`, потому что это служебные бинарные результаты ML-пайплайна, а не часть API-кода. Такое разделение оставляет рядом:

- код модели в `ml/model`
- model bundles для инференса в `ml/artifacts`

По умолчанию API и Docker используют `ml/artifacts/best_model.joblib`.

## Локальный запуск API

Установка зависимостей:

```bash
python -m pip install -r requirements.txt
```

Запуск через совместимый корневой entrypoint:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Канонический модульный запуск:
```bash
uvicorn apps.api.api:app --host 0.0.0.0 --port 8000
```

Запуск через Docker Compose:
```bash
docker compose up --build
```

## Локальный запуск ML-пайплайна

Совместимый запуск:

```bash
python main.py
```
