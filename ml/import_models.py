"""
Import trained checkpoints produced by the Kaggle notebook.

Unpacks `reddigen-models.zip` into `ml/models/` so the FastAPI server stops
serving rule-based stubs and starts running real inference. The server picks
each checkpoint up lazily on the next request — no restart needed.

Usage:
    python ml/import_models.py ~/Downloads/reddigen-models.zip
    python ml/import_models.py ~/Downloads/reddigen-models.zip --force
    python ml/import_models.py /path/to/unzipped/folder

Verify afterwards:
    curl http://localhost:8000/
    -> every entry in "models_loaded" should read true
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

EXPECTED = ["intent", "relevance", "role", "sentiment", "reply"]

# A directory is only a usable checkpoint if it carries weights AND a config.
WEIGHT_FILES = {"model.safetensors", "pytorch_model.bin"}


def find_model_dirs(root: Path) -> dict[str, Path]:
    """
    Locate checkpoint directories under `root`.

    Handles both zip layouts — models/<name>/ and <name>/ at the top level —
    plus sentence-transformers checkpoints, which nest their weights inside
    a numbered submodule folder rather than at the root.
    """
    found: dict[str, Path] = {}
    for name in EXPECTED:
        for candidate in (root / "models" / name, root / name, root / "ml" / "models" / name):
            if not candidate.is_dir():
                continue
            has_weights = any(
                (candidate / w).exists() for w in WEIGHT_FILES
            ) or any(candidate.rglob("*.safetensors")) or any(candidate.rglob("*.bin"))
            if has_weights:
                found[name] = candidate
                break
    return found


def dir_size_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive", help="reddigen-models.zip, or an unzipped folder")
    ap.add_argument("--dest", default="ml/models", help="target directory")
    ap.add_argument("--force", action="store_true", help="overwrite existing checkpoints")
    args = ap.parse_args()

    src = Path(args.archive).expanduser()
    if not src.exists():
        print(f"error: {src} not found")
        return 1

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        if src.is_file():
            if not zipfile.is_zipfile(src):
                print(f"error: {src} is not a zip archive")
                return 1
            print(f"Extracting {src.name} ({src.stat().st_size/1e6:.1f} MB)…")
            with zipfile.ZipFile(src) as z:
                z.extractall(tmp)
            root = Path(tmp)
        else:
            root = src

        models = find_model_dirs(root)
        if not models:
            print("error: no checkpoint directories found in the archive.")
            print(f"       expected one of {EXPECTED} containing model weights.")
            return 1

        print(f"\nFound {len(models)} checkpoint(s):")
        for name, path in sorted(models.items()):
            print(f"  {name:12} {dir_size_mb(path):8.1f} MB")

        installed, skipped = [], []
        for name, path in sorted(models.items()):
            target = dest / name
            if target.exists() and any(target.iterdir()):
                if not args.force:
                    skipped.append(name)
                    continue
                shutil.rmtree(target)
            # checkpoint-*/ holds optimizer + scheduler state from training.
            # Inference never reads it and it is several times the model size.
            shutil.copytree(path, target, ignore=shutil.ignore_patterns("checkpoint-*"))
            installed.append(name)

        print()
        for name in installed:
            print(f"  installed  {dest / name}")
        for name in skipped:
            print(f"  skipped    {dest / name} already exists (use --force to replace)")

        # Report only what is genuinely absent. A partial archive (e.g. one
        # that retrains two models) is normal, and the models it omits may
        # already be installed from an earlier import.
        absent = [m for m in EXPECTED if m not in models]
        already, stubbed = [], []
        for name in absent:
            target = dest / name
            (already if target.is_dir() and any(target.iterdir()) else stubbed).append(name)

        if already:
            print(f"\nnot in archive but already installed: {', '.join(already)} "
                  f"— left untouched.")
        if stubbed:
            print(f"\nnot installed: {', '.join(stubbed)} "
                  f"— those endpoints use rule-based stubs.")

    print("\nDone. Restart is not required — the server loads checkpoints lazily.")
    print("Check with:  curl http://localhost:8000/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
