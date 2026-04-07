# Deployment и CI/CD

## Docker

- Канонический Dockerfile API: `apps/api/Dockerfile`
- Compose для локального запуска: `docker-compose.yml`
- Совместимый корневой `Dockerfile` сохранён для старых сценариев сборки

API-контейнер использует `MODEL_PATH=/app/ml/artifacts/best_model.joblib` по умолчанию.

## GitHub Actions

Workflow `.github/workflows/ghcr.yml` использует явный `matrix` по контейнеризуемым модулям. Для MVP это выбрано вместо автодискавери, потому что:

- конфигурация читается проще
- добавление нового модуля требует одной записи в matrix
- меньше риск скрытой магии в CI

Сейчас настроен модуль:

- `api` -> `apps/api/Dockerfile`

## Как добавить новый модуль в CI

1. Создать директорию модуля, например `apps/ui`.
2. Добавить Dockerfile модуля.
3. Добавить запись в `matrix.include` workflow.
4. При необходимости расширить `on.push.paths`, если модуль использует новые manifest-файлы.
