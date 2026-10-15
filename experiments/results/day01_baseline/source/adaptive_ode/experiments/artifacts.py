"""Run directories with configuration, environment, and recoverable source."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from threadpoolctl import threadpool_info


REPOSITORY = Path(__file__).resolve().parents[2]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _command(arguments):
    return subprocess.check_output(arguments, cwd=REPOSITORY, text=True).strip()


def create_run(output, config):
    """Refuse to overwrite runs; snapshot code even when it is uncommitted."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    manifest = {}
    sources = sorted((REPOSITORY / "adaptive_ode").rglob("*.py"))
    sources += sorted((REPOSITORY / "tests").glob("*.py"))
    sources += sorted((REPOSITORY / "experiments" / "configs").glob("*.json"))
    sources += [REPOSITORY / "requirements-research.lock"]
    sources += [REPOSITORY / "docs" / "experiment_protocol.md", REPOSITORY / "docs" / "day01_audit.md"]
    for source in sources:
        if not source.is_file():
            continue
        relative = source.relative_to(REPOSITORY)
        data = source.read_bytes()
        manifest[str(relative)] = hashlib.sha256(data).hexdigest()
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    write_json(output / "source_manifest.json", manifest)
    canonical = json.dumps(config, sort_keys=True, allow_nan=False).encode()
    try:
        commit = _command(["git", "rev-parse", "HEAD"])
        git_status = _command(["git", "status", "--short"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, git_status = None, "Git metadata unavailable; use source snapshot."
    environment = _command([sys.executable, "-m", "pip", "freeze"])
    (output / "environment.txt").write_text(environment + "\n")
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git_commit": commit,
        "git_status": git_status,
        "config_sha256": hashlib.sha256(canonical).hexdigest(),
        "command": sys.argv,
        "threadpools": threadpool_info(),
        "status": "running",
    }
    write_json(output / "provenance.json", metadata)
    return output, metadata


def finish_run(output, metadata, elapsed_seconds, **details):
    metadata.update(
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=elapsed_seconds, status="complete", **details,
    )
    write_json(Path(output) / "provenance.json", metadata)
