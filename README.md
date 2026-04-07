# MVP-платформа анализа недвижимости

Репозиторий содержит текущий MVP для proxy valuation недвижимости и подготовлен к постепенному росту в сторону нескольких модулей внутри одного монорепозитория. Сейчас в репозитории уже выделены отдельный API-модуль и ML-часть; позже рядом можно добавлять `apps/ui` и другие сервисы без смены базовой структуры.

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
│   ├── data/raw/            # локальная копия датасета
│   ├── model/               # training/inference код
│   └── reports/             # generated reports и shortlist
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
- локальные данные в `ml/data/raw`
- generated reports в `ml/reports`
- model bundles для инференса в `ml/artifacts`

По умолчанию API и Docker используют `ml/artifacts/best_model.joblib`. Если артефакты станут слишком тяжёлыми для хранения в Git, структура уже готова для переноса их во внешний storage без перемешивания с кодом.

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

Канонический запуск:

```bash
python -m ml.model.main
```

## CI/CD

Workflow [`ghcr.yml`](./.github/workflows/ghcr.yml) собирает и публикует контейнеры модулей через явный `matrix`. Сейчас в matrix добавлен только `api`, но структура готова для новых модулей.

Что делает workflow:

- запускается только на релевантные изменения в `apps/**`, `ml/**`, workflow-файлах, `docker-compose.yml`, `requirements.txt`, `Dockerfile`
- билдит каждый модуль в своей matrix-ветке параллельно
- пушит образы в GHCR параллельно
- использует отдельный тег образа на модуль, например `ghcr.io/<owner>/<repo>-api`

Документация по архитектуре, API, ML и деплою лежит в [`docs/`](./docs).
