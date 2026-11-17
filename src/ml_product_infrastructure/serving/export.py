"""Экспорт чекпоинта ДЗ №3 в ONNX и проверка логитов перед публикацией файлов."""

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime as ort
import torch

from ml_product_infrastructure.dataset.files import CLASSES
from ml_product_infrastructure.dataset.s3 import S3Store
from ml_product_infrastructure.experiments.config import RunConfig, mapping
from ml_product_infrastructure.experiments.models import Classifier
from ml_product_infrastructure.serving.source import (
    Source,
    checked_download,
    download,
    manifest,
    parser,
)


def main() -> None:
    cli = parser(__doc__ or "")
    cli.add_argument("--repository", type=Path, default=Path("model_repository"))
    cli.add_argument("--stats", type=Path, default=Path("artifacts/serving/stats.json"))
    args = cli.parse_args()
    repository: Path = args.repository
    stats: Path = args.stats
    source = Source.read(args.config)
    store = S3Store.from_env()
    hashes, digest = manifest(store, source)
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(42)

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoint_path = root / "model.pt"
        download(store, source.checkpoint, checkpoint_path)
        checkpoint = mapping(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        metadata = mapping(checkpoint["metadata"])
        config = RunConfig.from_dict(metadata["config"])
        if (
            config.data.s3_path != source.dataset_prefix
            or metadata["manifest_sha256"] != digest
            or metadata["classes"] != CLASSES
        ):
            raise ValueError("Чекпоинт относится к другой версии данных или набору классов")
        state = mapping(checkpoint["model_state_dict"])
        model = Classifier(config.model).eval()
        model.load_state_dict({key: torch.as_tensor(value) for key, value in state.items()})
        exported = root / "model.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                model,
                (torch.randn(2, 3, 32, 32, generator=generator),),
                exported,
                dynamo=False,
                opset_version=17,
                external_data=False,
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            )
            onnx.checker.check_model(str(exported))
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(
                str(exported), sess_options=options, providers=["CPUExecutionProvider"]
            )
            for batch in (1, 2, 64):
                inputs = torch.randn(batch, 3, 32, 32, generator=generator)
                actual = session.run(["logits"], {"input": inputs.numpy()})[0]
                expected = model(inputs).numpy()
                np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
                print(
                    f"ONNX Runtime · batch {batch} · max |Δ| = {np.max(abs(actual - expected)):.3g}"
                )
        checked_download(
            store, source.dataset + "/stats.json", root / "stats.json", hashes["stats.json"]
        )
        model_folder = repository / "cifar_cnn"
        (model_folder / "1").mkdir(parents=True, exist_ok=True)
        stats.parent.mkdir(parents=True, exist_ok=True)
        for original, destination in (
            (exported, model_folder / "1/model.onnx"),
            (Path("configs/triton.pbtxt"), model_folder / "config.pbtxt"),
            (root / "stats.json", stats),
        ):
            staged = destination.with_name(destination.name + ".tmp")
            shutil.copyfile(original, staged)
            staged.replace(destination)
    print(f"Готово: {repository}; stats: {stats}")


if __name__ == "__main__":
    main()
