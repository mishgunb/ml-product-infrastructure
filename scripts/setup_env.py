"""Добавить новые настройки из примера, сохранив существующие значения .env."""

from pathlib import Path

from dotenv import dotenv_values


def main() -> None:
    path = Path(".env")
    if not path.exists():
        path.write_text(Path(".env.example").read_text())
        return
    current = dotenv_values(path)
    missing = [
        (key, value)
        for key, value in dotenv_values(".env.example").items()
        if key not in current and value is not None
    ]
    if missing:
        with path.open("a") as stream:
            stream.write("\n# Новые настройки проекта\n")
            for key, value in missing:
                stream.write(f"{key}={value}\n")


if __name__ == "__main__":
    main()
