"""
Shared MLflow setup for the five training scripts.

Every training run records: hyperparameters, dataset fingerprint (size and
class balance), per-epoch metrics, the final evaluation, and the exported
checkpoint as an artifact. Runs are grouped under one experiment per model
so they can be compared across backbones and hyperparameters.

Tracking backend:
  - default: a local file store at ./mlruns (works offline, including on
    Kaggle where there is no tracking server)
  - override with MLFLOW_TRACKING_URI for a remote server or DagsHub

Inspect results locally with:
    mlflow ui --backend-store-uri ./mlruns
"""

from __future__ import annotations

import os
import platform
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:  # training must still work without mlflow installed
    mlflow = None  # type: ignore
    MLFLOW_AVAILABLE = False


# MLflow 3.x put the plain-file store into maintenance mode and raises unless
# you opt out, so the default is a local SQLite database. It needs no server,
# is a single portable file, and is what `mlflow ui` expects.
DEFAULT_STORE = "sqlite:///mlflow.db"


def init(experiment: str) -> bool:
    """
    Point MLflow at a tracking store and select the experiment.

    Returns True when tracking is active. Callers should degrade gracefully
    when it is not, so a missing mlflow install never blocks training.
    """
    if not MLFLOW_AVAILABLE:
        print("[mlflow] not installed — skipping experiment tracking")
        return False

    uri = _normalise_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_STORE))
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    print(f"[mlflow] tracking to {uri} (experiment: {experiment})")
    return True


def _normalise_uri(uri: str) -> str:
    """
    Turn a filesystem path into a file:// URI.

    MLflow's registry rejects bare paths — a Windows path like
    C:\\runs\\mlruns is read as scheme "c", which is unsupported. Anything
    that already carries a scheme (http, sqlite, databricks, file) passes
    through untouched.
    """
    if "://" not in uri:
        uri = Path(uri).expanduser().resolve().as_uri()
    if uri.startswith("file:"):
        # Opt back into the deprecated file store rather than failing outright,
        # for anyone who has pointed MLFLOW_TRACKING_URI at a directory.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    return uri


def log_environment() -> None:
    """Record the hardware/software context — needed to explain timings."""
    if not MLFLOW_AVAILABLE:
        return
    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        pass
    mlflow.log_params({f"env.{k}": v for k, v in info.items()})


def log_dataset(rows: list[dict], label_key: str | None = None) -> None:
    """Log dataset size and, when labelled, the class balance."""
    if not MLFLOW_AVAILABLE:
        return
    mlflow.log_param("dataset.rows", len(rows))
    if label_key:
        counts = Counter(r[label_key] for r in rows if label_key in r)
        for label, n in sorted(counts.items()):
            mlflow.log_param(f"dataset.count.{label}", n)
            mlflow.log_metric(f"dataset.share.{label}", n / max(1, len(rows)))


def log_checkpoint(path: str | Path, artifact_path: str = "model") -> None:
    """
    Attach the exported checkpoint directory to the run.

    The five checkpoints total roughly 2.5 GB, and copying them into the
    MLflow artifact store doubles that. Set MLFLOW_LOG_CHECKPOINTS=false to
    record only the size metric — useful on Kaggle, where the working
    directory has a quota and the checkpoints are exported separately.
    """
    if not MLFLOW_AVAILABLE:
        return
    p = Path(path)
    if not p.exists():
        print(f"[mlflow] checkpoint {p} missing — not logged")
        return

    size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
    mlflow.log_metric("checkpoint.size_mb", round(size_mb, 2))

    if os.environ.get("MLFLOW_LOG_CHECKPOINTS", "true").lower() in ("0", "false", "no"):
        print(f"[mlflow] checkpoint artifact upload disabled ({size_mb:.1f} MB not copied)")
        return

    mlflow.log_artifacts(str(p), artifact_path=artifact_path)
    print(f"[mlflow] logged checkpoint ({size_mb:.1f} MB)")


def report_to() -> list[str]:
    """Value for HuggingFace TrainingArguments.report_to."""
    return ["mlflow"] if MLFLOW_AVAILABLE else []


class run:
    """
    Context manager wrapping mlflow.start_run, inert when mlflow is absent.

        with mlflow_utils.run("intent-distilbert", params):
            ...
    """

    def __init__(self, name: str, params: dict[str, Any] | None = None):
        self.name = name
        self.params = params or {}
        self.active = False

    def __enter__(self):
        if MLFLOW_AVAILABLE:
            mlflow.start_run(run_name=self.name)
            mlflow.log_params(self.params)
            log_environment()
            self.active = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.active:
            if exc_type is not None:
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("error", str(exc)[:500])
            else:
                mlflow.set_tag("status", "finished")
            mlflow.end_run()
        return False  # never swallow exceptions
