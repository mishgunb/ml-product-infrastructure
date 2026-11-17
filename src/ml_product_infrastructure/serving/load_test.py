"""Замкнутая нагрузка реальными PNG, с отдельным прогревом каждого сценария."""

import argparse
import csv
import json
import os
import platform
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier
from time import perf_counter

import httpx
import numpy as np

from ml_product_infrastructure.dataset.s3 import S3Store
from ml_product_infrastructure.experiments.config import mapping
from ml_product_infrastructure.serving.source import Source, fetch_test_images, parser


@dataclass(frozen=True)
class Measurement:
    request: int
    worker: int
    ms: float
    status: str


def measure(
    client: httpx.Client, images: list[bytes], workers: int
) -> tuple[list[Measurement], float]:
    start = Barrier(workers + 1)

    def run(worker: int) -> list[Measurement]:
        results: list[Measurement] = []
        start.wait()
        for index in range(worker, len(images), workers):
            begin = perf_counter()
            try:
                response = client.post(
                    "/predict", files={"file": ("test.png", images[index], "image/png")}
                )
                status = str(response.status_code)
            except httpx.RequestError as exc:
                status = type(exc).__name__
            results.append(
                Measurement(index + 1, worker + 1, (perf_counter() - begin) * 1000, status)
            )
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(run, worker) for worker in range(workers)]
        begin = perf_counter()
        start.wait()
        measurements = [item for job in jobs for item in job.result()]
        elapsed = perf_counter() - begin
    return sorted(measurements, key=lambda item: item.request), elapsed


def client_cpu() -> str:
    if platform.system() == "Darwin":
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.partition(":")[0].strip() in {"model name", "Hardware"}:
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine()


def counter(metrics: str, name: str, model: str) -> float:
    return sum(
        float(line.rsplit(maxsplit=1)[1])
        for line in metrics.splitlines()
        if line.startswith(f"{name}{{") and f'model="{model}"' in line
    )


def conclusions(summary: list[tuple[int, float, float, float]]) -> str:
    first, last = summary[0], summary[-1]
    peak = max(summary, key=lambda row: row[1])
    plateau = next(
        (
            row[0]
            for row in summary[:-1]
            if all(later[1] <= row[1] * 1.05 for later in summary if later[0] > row[0])
        ),
        None,
    )
    trend = (
        f"После {plateau} workers прирост RPS оставался в пределах 5% либо RPS снижался."
        if plateau is not None
        else "До максимального числа workers устойчивого плато по порогу 5% не наблюдалось."
    )
    return (
        f"При переходе с {first[0]} до {last[0]} workers RPS изменился с {first[1]:.1f} "
        f"до {last[1]:.1f}, а p95 — с {first[2]:.2f} до {last[2]:.2f} мс. "
        f"Максимум составил {peak[1]:.1f} RPS при {peak[0]} workers. {trend} "
        f"Средний батч Triton изменился с {first[3]:.2f} до {last[3]:.2f}; "
        "после загрузки доступных ресурсов дополнительные запросы увеличивают конкуренцию "
        "за CPU и ожидание, а измеренный RPS включает и обработку HTTP-обёрткой."
    )


def main() -> None:
    cli = parser(__doc__ or "")
    cli.add_argument("--url", default="http://127.0.0.1:8080")
    cli.add_argument("--triton-url", default="http://127.0.0.1:8000")
    cli.add_argument("--metrics-url", default="http://127.0.0.1:8002/metrics")
    cli.add_argument("--model", default="cifar_cnn")
    cli.add_argument("--cache", type=Path, default=Path("data/serving"))
    cli.add_argument("--report", type=Path, default=Path("reports/load_test.md"))
    cli.add_argument("--raw", type=Path, default=Path("reports/load_test.csv"))
    cli.add_argument("--workers", nargs="+", type=int, default=[1, 2, 5, 10])
    cli.add_argument("--requests", type=int, default=1000)
    cli.add_argument("--warmup", type=int, default=50)
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument(
        "--service-host", required=True, help="ОС/архитектура, ядра и размещение сервиса"
    )
    args: argparse.Namespace = cli.parse_args()
    workers = sorted(set(args.workers))
    if min(workers) < 1 or args.requests < max(1000, max(workers)) or args.warmup < 1:
        cli.error("Нужны workers ≥ 1, минимум 1000 запросов и хотя бы один запрос прогрева")
    source = Source.read(args.config)
    rows, root, _ = fetch_test_images(S3Store.from_env(), source, args.cache)
    rng = random.Random(args.seed)
    warmup = [(root / row["file_name"]).read_bytes() for row in rng.choices(rows, k=args.warmup)]
    images = [(root / row["file_name"]).read_bytes() for row in rng.choices(rows, k=args.requests)]
    table = [
        "| Workers | Запросов | Ошибок | RPS | Среднее, мс | Медиана, мс | p90, мс | p95, мс |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    summary: list[tuple[int, float, float, float]] = []
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=10) as client, args.raw.open("w", newline="") as raw:
        client.get(args.url.rstrip("/") + "/health").raise_for_status()
        server = mapping(client.get(args.triton_url + "/v2").raise_for_status().json())
        route = f"/v2/models/{args.model}/config"
        config = mapping(client.get(args.triton_url + route).raise_for_status().json())
        writer = csv.writer(raw)
        writer.writerow(["workers", "request", "worker", "latency_ms", "status"])
        for count in workers:
            with httpx.Client(base_url=args.url, timeout=10) as predict_client:
                measure(predict_client, warmup, count)
                before = client.get(args.metrics_url).raise_for_status().text
                samples, elapsed = measure(predict_client, images, count)
                after = client.get(args.metrics_url).raise_for_status().text
            for sample in samples:
                writer.writerow(
                    [count, sample.request, sample.worker, f"{sample.ms:.6f}", sample.status]
                )
            success = [sample.ms for sample in samples if sample.status == "200"]
            if not success:
                raise SystemExit(
                    f"Все запросы завершились ошибкой при workers={count}; см. {args.raw}"
                )
            latencies = [sample.ms for sample in samples]
            median, p90, p95 = np.percentile(latencies, [50, 90, 95])
            rps = len(success) / elapsed
            table.append(
                f"| {count} | {len(samples)} | {len(samples) - len(success)} | {rps:.1f} | "
                f"{np.mean(latencies):.2f} | {median:.2f} | {p90:.2f} | {p95:.2f} |"
            )
            print(table[-1], flush=True)
            inferences, executions = (
                counter(after, key, args.model) - counter(before, key, args.model)
                for key in ("nv_inference_count", "nv_inference_exec_count")
            )
            summary.append(
                (count, rps, float(p95), inferences / executions if executions else 0.0)
            )

    conditions = [
        f"- Клиент: {platform.platform()}, {client_cpu()}, {os.cpu_count()} логических CPU.",
        f"- Сервис: {args.service_host}.",
        f"- Triton {server['version']}; модель `{args.model}`, max_batch_size={config['max_batch_size']}.",
        f"- Экземпляры: `{json.dumps(config['instance_group'])}`.",
        f"- Dynamic batching: `{json.dumps(config['dynamic_batching'])}`.",
        f"- {args.warmup} запросов прогрева и {args.requests} измеряемых запросов на сценарий, seed={args.seed}.",
        "- Задержки по всем попыткам; RPS по HTTP 200. Изображения загружались в память до замеров.",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# Нагрузочный тест POST /predict\n\n"
        + "\n".join(table)
        + "\n\n## Условия\n\n"
        + "\n".join(conditions)
        + "\n\n## Выводы\n\n"
        + conclusions(summary)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
