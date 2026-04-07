# MVP Proxy Valuation для недвижимости

## Что это

Этот проект реализует MVP valuation core для shortlist вероятно недооцененных объектов недвижимости.

Основная логика:

1. На вход подаются характеристики объекта.
2. Модель оценивает `expected_price_proxy`.
3. Если известна цена объявления, система сравнивает ее с модельной оценкой.
4. Считаются:
   - `delta_abs = expected_price_proxy - listing_price`
   - `delta_pct = delta_abs / listing_price`
5. Объекты можно ранжировать по вероятной недооцененности.

## Важное ограничение MVP

Модель обучена на данных объявлений, а не на реальных сделках.

Из этого следуют три принципиальных ограничения:

- результат называется `expected_price_proxy`, а не fair market value;
- это proxy valuation на основе listing data;
- модель подходит для shortlist, ranking и первичной аналитики, но не для финальной оценки рыночной стоимости.

В ответах API это отдельно фиксируется через `valuation_note`.

## Данные

Источник:

- Hugging Face dataset `raimbekovm/bishkek-real-estate`
- локальная копия: `data/raw/listings.csv`

Таргет:

- `price_usd`

Базовая валюта модели:

- `USD`

## Что использует модель

Реальные признаки, ожидаемые сохраненным артефактом модели:

Числовые:

- `rooms`
- `total_area_m2`
- `living_area_m2`
- `kitchen_area_m2`
- `floor`
- `total_floors`
- `ceiling_height`
- `year_built`
- `latitude`
- `longitude`
- `photo_count`
- `area_per_room` `auto`
- `floor_ratio` `auto`
- `building_age` `auto`
- `is_top_floor` `auto`
- `is_first_floor` `auto`
- `has_coordinates` `auto`

Категориальные:

- `building_type`
- `building_series`
- `condition`
- `heating`
- `gas_supply`
- `bathroom`
- `balcony`
- `parking`
- `furniture`
- `flooring`
- `door_type`
- `has_landline_phone`
- `internet`
- `mortgage`
- `seller_type`
- `district`

Не используются:

- `price_per_m2_usd` как leakage
- `listing_id`, `url`, `city`, `address`, `description`, `parsed_at`, `photos_downloaded`
- текстовые поля `amenities`, `documents`, `security`

## Обучение модели

В проекте реализованы две модели:

1. `LinearRegression` как baseline
2. `CatBoostRegressor` как основная модель

Подход:

- очистка данных и базовый quality control;
- подготовка признаков;
- train/validation split;
- обучение на `log1p(price_usd)`;
- обратное преобразование через `expm1` на inference;
- сравнение моделей по `RMSE`, `MAE`, `R2`, `MAPE`.

Выбранная лучшая модель:

- `CatBoostRegressor`

Validation metrics:

- `MAE = 12429.54`
- `RMSE = 21672.48`
- `R2 = 0.9195`
- `MAPE = 0.0926`

## Что возвращает inference

Базовый результат:

- `expected_price_proxy`

Если передан `listing_price` или `price_usd`, дополнительно считаются:

- `listing_price`
- `delta_abs`
- `delta_pct`

Формулы:

- `delta_abs = expected_price_proxy - listing_price`
- `delta_pct = delta_abs / listing_price`

Интерпретация:

- `delta_pct > 0` и `delta_abs > 0` — объект выглядит недооцененным относительно proxy-модели;
- `delta_pct < 0` и `delta_abs < 0` — объект выглядит переоцененным относительно proxy-модели.

## Поддержка валют

Модель предсказывает в базовой валюте `USD`.

Поверх модели реализован post-processing слой:

- `output_currency = "USD"`
- `output_currency = "RUB"`
- `output_currency = "BOTH"`

Если требуется RUB:

- сначала считается `expected_price_proxy` в USD;
- затем денежные поля конвертируются через `fx_rate`;
- `delta_pct` не меняется;
- в ответе возвращается `fx_rate_used`.

Если `fx_rate` не передан, используется fallback из конфигурации.

## Explainability

Для краткого объяснения результата добавлен легковесный explainability-слой.

Текущая логика:

- для `CatBoostRegressor` используется локальная SHAP-подобная интерпретация через встроенный механизм CatBoost;
- если локальная интерпретация недоступна, используется fallback на простые эвристики.

В ответе доступны:

- `top_factors` — 3-5 кратких факторов, которые повысили или снизили proxy-оценку;
- `explanation_summary` — короткое human-readable объяснение;
- `valuation_note` — напоминание, что это proxy valuation на listing data.

## API

В проекте уже есть FastAPI-слой:

- [api.py](/Users/daniel/Projects/ДИПЛОМ/модель/api.py)

Основные ручки:

- `GET /health`
- `POST /predict`
- `POST /predict/batch`

Ключевые возможности API:

- одиночный и batch inference;
- расчет `expected_price_proxy`, `delta_abs`, `delta_pct`;
- возврат в `USD`, `RUB` или `BOTH`;
- `fx_rate_used` в ответе;
- объяснимость через `top_factors` и `explanation_summary`;
- batch ranking по вероятной недооцененности.

Подробный контракт API:

- [API_REFERENCE.md](/Users/daniel/Projects/ДИПЛОМ/модель/API_REFERENCE.md)

## Структура проекта

- `main.py` — training pipeline и генерация артефактов
- `preprocessing.py` — очистка данных и feature engineering
- `train.py` — обучение baseline и CatBoost
- `evaluate.py` — метрики и отчеты
- `inference.py` — загрузка модели, inference, валютный post-processing, explainability
- `api.py` — FastAPI API
- `artifacts/` — сохраненные модели
- `reports/` — отчеты, shortlist, feature importance

## Артефакты

Основные файлы:

- `artifacts/best_model.joblib`
- `artifacts/catboost_regressor.joblib`
- `artifacts/linear_regression_baseline.joblib`
- `reports/model_comparison.csv`
- `reports/ranked_undervalued_listings.csv`
- `reports/shortlist_top_100.csv`

## Установка

```bash
python -m pip install -r requirements.txt
```

## Обучение

```bash
python main.py
```

## Запуск API

Стандартный запуск:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Запуск с fallback-курсом для RUB:

```bash
DEFAULT_FX_RATE=90 uvicorn api:app --host 0.0.0.0 --port 8000
```

## Для чего проект подходит

- shortlist вероятно недооцененных объявлений;
- приоритизация объектов для ручной проверки;
- ranking и аналитика внутри MVP;
- быстрая автоматизированная proxy-оценка новых листингов.

## Для чего проект не подходит

- для юридически значимой оценки;
- для transaction-based valuation;
- для заявления точной рыночной стоимости объекта;
- для замены полноценной appraisal / valuation экспертизы.
