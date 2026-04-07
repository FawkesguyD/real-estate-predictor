# API Reference

## Назначение API

API поднимает обученную MVP-модель как сервис для proxy valuation объектов недвижимости.

Это не transaction-based valuation API. Сервис возвращает модельную оценку `expected_price_proxy`, обученную на listing data, и помогает:

- быстро оценить объект по его характеристикам;
- сравнить модельную оценку с `listing_price`;
- посчитать отклонение в абсолютном и относительном выражении;
- построить shortlist вероятно недооцененных объявлений;
- получить краткое объяснение, какие признаки сильнее всего повлияли на результат.

## Честная интерпретация результата

Сервис обучен на `price_usd` из объявлений, а не на ценах реальных сделок.

Поэтому:

- `expected_price_proxy` — это proxy valuation;
- это не fair market value;
- результат предназначен для shortlist и ranking;
- финальная инвестиционная или оценочная интерпретация должна делаться отдельно.

Каждый ответ содержит `valuation_note` с этим ограничением.

## Базовые принципы

Базовая валюта модели:

- `USD`

Поддерживаемые режимы ответа:

- `output_currency = "USD"`
- `output_currency = "RUB"`
- `output_currency = "BOTH"`

Если объект содержит цену объявления, считаются:

- `listing_price`
- `delta_abs`
- `delta_pct`

Формулы:

- `delta_abs = expected_price_proxy - listing_price`
- `delta_pct = delta_abs / listing_price`

Важно:

- `delta_pct` возвращается как доля, а не как проценты;
- при конвертации валют меняются только денежные поля;
- `delta_pct` не меняется;
- при `RUB` и `BOTH` сервис всегда возвращает `fx_rate_used`.

## Формат ответа

Одиночный и batch response используют одну и ту же логику денежных полей.

Денежные значения сгруппированы в `price_outputs`:

```json
{
  "price_outputs": {
    "USD": {
      "expected_price_proxy": 128156.27,
      "listing_price": 130000.0,
      "delta_abs": -1843.73,
      "delta_pct": -0.01418
    },
    "RUB": {
      "expected_price_proxy": 11534064.58,
      "listing_price": 11700000.0,
      "delta_abs": -165935.42,
      "delta_pct": -0.01418
    }
  }
}
```

Общие поля:

- `base_currency`
- `output_currency`
- `fx_rate_used`
- `price_outputs`
- `valuation_note`

Опциональные explainability-поля:

- `top_factors`
- `explanation_summary`

## Endpoint: `GET /health`

Проверяет, что API доступен и артефакт модели загружается.

Пример:

```bash
curl http://127.0.0.1:8000/health
```

Ответ:

```json
{
  "status": "ok"
}
```

## Endpoint: `POST /predict`

### Назначение

Оценивает один объект.

### Request body

```json
{
  "object_features": {
    "price_usd": 130000,
    "rooms": 3,
    "total_area_m2": 90.8,
    "living_area_m2": 90.0,
    "kitchen_area_m2": 30.0,
    "floor": 4,
    "total_floors": 4,
    "building_type": "кирпич",
    "building_series": "индивид. планировка",
    "year_built": 2011,
    "condition": "хорошее",
    "district": "11 м-н",
    "latitude": 42.8228,
    "longitude": 74.6334,
    "photo_count": 14
  },
  "output_currency": "BOTH",
  "fx_rate": 89.5,
  "include_explanation": true
}
```

### Параметры

- `object_features` — словарь признаков объекта
- `output_currency` — `USD`, `RUB` или `BOTH`
- `fx_rate` — опциональный курс USD/RUB
- `include_explanation` — включать ли explainability-блок

### Допустимые имена цены объявления

Предпочтительный вариант:

- `price_usd`

Также поддерживается:

- `listing_price`

Если передано `listing_price`, сервис автоматически интерпретирует его как цену объявления в USD.

### Response example

```json
{
  "base_currency": "USD",
  "output_currency": "BOTH",
  "fx_rate_used": 89.5,
  "price_outputs": {
    "USD": {
      "expected_price_proxy": 128156.27308416489,
      "listing_price": 130000.0,
      "delta_abs": -1843.7269158351119,
      "delta_pct": -0.014182514737193168
    },
    "RUB": {
      "expected_price_proxy": 11469986.441032758,
      "listing_price": 11635000.0,
      "delta_abs": -165013.5589672425,
      "delta_pct": -0.014182514737193168
    }
  },
  "top_factors": [
    "Фактор «площадь 90.8 м²» повысил proxy-оценку",
    "Фактор «комнатность 3» повысил proxy-оценку",
    "Фактор «количество фото 14» повысил proxy-оценку",
    "Фактор «широта объекта» повысил proxy-оценку",
    "Фактор «соотношение площади к комнатам» снизил proxy-оценку"
  ],
  "explanation_summary": "Proxy-оценка в основном поддержана факторами: площадь 90.8 м², комнатность 3. При этом признаки вроде соотношение площади к комнатам частично снизили оценку. Это модельная оценка по данным объявлений, а не transaction-based fair market valuation.",
  "valuation_note": "MVP proxy valuation based on listing prices; not a transaction-based fair market valuation."
}
```

### Как читать ответ

- `price_outputs.USD.expected_price_proxy` — proxy-оценка модели в базовой валюте
- `price_outputs.USD.listing_price` — цена объявления в USD
- `price_outputs.USD.delta_abs` — абсолютная разница между моделью и объявлением
- `price_outputs.USD.delta_pct` — относительная разница
- `price_outputs.RUB.*` — те же денежные поля после конвертации
- `top_factors` — 3-5 наиболее заметных факторов
- `explanation_summary` — короткое человекочитаемое объяснение

### Если `listing_price` не передан

API все равно вернет `expected_price_proxy`.

Тогда:

- `listing_price = null`
- `delta_abs = null`
- `delta_pct = null`

## Endpoint: `POST /predict/batch`

### Назначение

Оценивает список объектов и при необходимости ранжирует их по вероятной недооцененности.

### Request body

```json
{
  "objects": [
    {
      "listing_id": "obj-1",
      "price_usd": 130000,
      "rooms": 3,
      "total_area_m2": 90.8,
      "district": "11 м-н",
      "latitude": 42.8228,
      "longitude": 74.6334
    },
    {
      "listing_id": "obj-2",
      "price_usd": 120000,
      "rooms": 3,
      "total_area_m2": 85,
      "district": "12 м-н",
      "latitude": 42.84,
      "longitude": 74.60
    }
  ],
  "rank_by_undervaluation": true,
  "output_currency": "USD",
  "fx_rate": null,
  "include_explanations": false
}
```

### Параметры

- `objects` — список объектов
- `rank_by_undervaluation` — сортировать ли результат по недооцененности
- `output_currency` — `USD`, `RUB` или `BOTH`
- `fx_rate` — опциональный курс USD/RUB
- `include_explanations` — считать ли explainability для каждого объекта

### Response example

```json
{
  "count": 2,
  "ranked": true,
  "results": [
    {
      "input_index": 0,
      "listing_id": "obj-1",
      "base_currency": "USD",
      "output_currency": "USD",
      "fx_rate_used": null,
      "price_outputs": {
        "USD": {
          "expected_price_proxy": 128156.27308416489,
          "listing_price": 130000.0,
          "delta_abs": -1843.7269158351119,
          "delta_pct": -0.014182514737193168
        }
      },
      "top_factors": [],
      "explanation_summary": null,
      "valuation_note": "MVP proxy valuation based on listing prices; not a transaction-based fair market valuation.",
      "undervaluation_rank": 1
    },
    {
      "input_index": 1,
      "listing_id": "obj-2",
      "base_currency": "USD",
      "output_currency": "USD",
      "fx_rate_used": null,
      "price_outputs": {
        "USD": {
          "expected_price_proxy": 106558.74300318903,
          "listing_price": 120000.0,
          "delta_abs": -13441.256996810975,
          "delta_pct": -0.11201047497342478
        }
      },
      "top_factors": [],
      "explanation_summary": null,
      "valuation_note": "MVP proxy valuation based on listing prices; not a transaction-based fair market valuation.",
      "undervaluation_rank": 2
    }
  ]
}
```

### Как читать batch-ответ

- `input_index` — индекс объекта во входном массиве
- `listing_id` — id объявления, если был передан
- `undervaluation_rank` — позиция в ранжировании
- `count` — число объектов в ответе
- `ranked` — был ли применен ranking

### Как интерпретировать ranking

- `delta_pct > 0` и `delta_abs > 0` — объект выглядит недооцененным относительно proxy-модели
- `delta_pct < 0` и `delta_abs < 0` — объект выглядит переоцененным относительно proxy-модели
- более высокий `undervaluation_rank` означает более высокий приоритет для ручной проверки

## Explainability

Explainability в текущем MVP легковесная и не требует переобучения модели.

Текущая логика:

- для `CatBoostRegressor` используется локальная SHAP-подобная интерпретация через встроенный механизм CatBoost;
- если это недоступно, используется fallback на эвристики.

Что возвращается:

- `top_factors`
- `explanation_summary`

Это объяснение нужно воспринимать как supporting context для shortlist, а не как строгую причинную интерпретацию.

## Реальные признаки модели

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
- `area_per_room`
- `floor_ratio`
- `building_age`
- `is_top_floor`
- `is_first_floor`
- `has_coordinates`

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

Производные признаки строятся автоматически внутри preprocessing/inference pipeline.

## Запуск API

Стандартно:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

С fallback-курсом для RUB:

```bash
DEFAULT_FX_RATE=90 uvicorn api:app --host 0.0.0.0 --port 8000
```
