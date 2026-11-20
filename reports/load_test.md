# Нагрузочный тест POST /predict

| Workers | Запросов | Ошибок | RPS | Среднее, мс | Медиана, мс | p90, мс | p95, мс |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1000 | 0 | 22.8 | 43.76 | 34.18 | 49.85 | 64.02 |
| 2 | 1000 | 0 | 40.6 | 49.20 | 48.46 | 57.24 | 61.91 |
| 5 | 1000 | 0 | 71.1 | 67.95 | 66.33 | 95.83 | 107.65 |
| 10 | 1000 | 0 | 73.2 | 132.72 | 131.26 | 202.84 | 219.75 |

## Условия

- Клиент: macOS-26.6.2-arm64-arm-64bit, Apple M1, 8 логических CPU.
- Сервис: macOS 26.6.2, Apple M1 (8 ядер), Colima Linux ARM64 с 4 vCPU и 4 GiB RAM, Triton CPU x86_64 под эмуляцией, клиент и сервис на одном Mac, образ giang352001/tritonserverbuild:25.11-cpu sha256:65e1d25bff51e5b7c39fb58c1e356bc58c6b91ac729914ad1483a783c7eafa9d.
- Triton 2.63.0; модель `cifar_cnn`, max_batch_size=64.
- Экземпляры: `[{"name": "cifar_cnn_0", "kind": "KIND_CPU", "count": 2, "gpus": [], "secondary_devices": [], "profile": [], "passive": false, "host_policy": ""}]`.
- Dynamic batching: `{"preferred_batch_size": [64], "max_queue_delay_microseconds": 500, "preserve_ordering": false, "priority_levels": 0, "default_priority_level": 0, "priority_queue_policy": {}}`.
- 50 запросов прогрева и 1000 измеряемых запросов на сценарий, seed=42.
- Задержки по всем попыткам; RPS по HTTP 200. Изображения загружались в память до замеров.

## Выводы

При переходе с 1 до 10 workers RPS изменился с 22.8 до 73.2, а p95 — с 64.02 до 219.75 мс. Максимум составил 73.2 RPS при 10 workers. После 5 workers прирост RPS оставался в пределах 5% либо RPS снижался. Средний батч Triton изменился с 1.00 до 2.31; после загрузки доступных ресурсов дополнительные запросы увеличивают конкуренцию за CPU и ожидание, а измеренный RPS включает и обработку HTTP-обёрткой.
