#!/bin/sh
# Используем установленный uv либо устанавливаем фиксированную версию локально.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export UV_CACHE_DIR="$ROOT/.cache/uv"
if command -v uv >/dev/null 2>&1; then
    exec uv "$@"
fi
UV_LOCAL="$ROOT/.tools/uv"
if [ ! -x "$UV_LOCAL/uv" ]; then
    mkdir -p "$UV_LOCAL"
    INSTALLER=$(mktemp)
    trap 'rm -f "$INSTALLER"' EXIT HUP INT TERM
    curl --fail --silent --show-error --location https://astral.sh/uv/0.9.17/install.sh -o "$INSTALLER"
    UV_INSTALL_DIR="$UV_LOCAL" UV_NO_MODIFY_PATH=1 sh "$INSTALLER"
fi
exec "$UV_LOCAL/uv" "$@"
