"""Классы и пять проверенных примеров test для веб-демо."""

from pathlib import Path
from tempfile import TemporaryDirectory

from ml_product_infrastructure.dataset.files import read_annotation, write_json
from ml_product_infrastructure.dataset.s3 import S3Store
from ml_product_infrastructure.serving.source import Source, checked_download, manifest, parser


def prepare(config: Path, output: Path) -> None:
    source = Source.read(config)
    store = S3Store.from_env()
    hashes, digest = manifest(store, source)
    with TemporaryDirectory() as temporary:
        annotation_path = Path(temporary) / "test.json"
        checked_download(
            store,
            source.dataset + "/annotations/test.json",
            annotation_path,
            hashes["annotations/test.json"],
        )
        annotation = read_annotation(annotation_path)
        selected: set[int] = set()
        examples: list[dict[str, str]] = []
        for row in annotation["images"]:
            if row["label"] in selected:
                continue
            selected.add(row["label"])
            image_id = row["image_id"]
            checked_download(
                store,
                source.dataset + "/" + row["file_name"],
                output / f"{image_id}.png",
                row["sha256"],
            )
            examples.append({"id": image_id, "class_name": row["label_name"]})
            if len(examples) == 5:
                break
    if len(examples) < 3:
        raise ValueError("В test нужны примеры хотя бы трёх разных классов")
    write_json(
        output / "assets.json",
        {
            "classes": annotation["classes"],
            "examples": examples,
            "dataset": source.dataset,
            "manifest_sha256": digest,
        },
    )
    print(f"Готово: {len(examples)} примеров разных классов из {source.dataset} → {output}")


def main() -> None:
    cli = parser(__doc__ or "")
    cli.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    args = cli.parse_args()
    prepare(args.config, args.output)


if __name__ == "__main__":
    main()
