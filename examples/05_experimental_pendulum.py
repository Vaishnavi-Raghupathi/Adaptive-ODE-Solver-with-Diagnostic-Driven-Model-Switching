"""Download and evaluate four measured free-swing double-pendulum recordings.

This is a data-driven prediction pilot, not an integrator accuracy benchmark.
The five-second horizon, file roles and accuracy target are fixed before scoring.
Raw upstream data stay in the user-supplied source directory, outside the package.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
from scipy.io import loadmat

from adaptive_ode import BenchmarkConfig, Trajectory, benchmark_trajectories


REPOSITORY = "https://github.com/dynamicslab/MultiArm-Pendulum"
COMMIT = "e99a7e13fe2adc8fc647321a156b6cb59a48a698"
FILES = {
    "DoubleDataFreeSwing_1_Dt_0_001.mat": "bbcd935907e4c91ef1058eccdc74aca0a8ed9b82",
    "DoubleDataFreeSwing_2_Dt_0_001.mat": "2ec0f34b260374942bbba4c64334abbfd157340d",
    "DoubleDataFreeSwing_3_Dt_0_001.mat": "57f82456c934a339fd1dbdf8b48c6167047bd8ae",
    "DoubleDataFreeSwing_4_Dt_0_001.mat": "30eca7c58b03450e38e2435071fa83f12bf6c02e",
    "README.md": "f8b768537166dcc703ddf86174f253298a349899",
    "LICENSE": "7e1f4887dc2d8437c1b2dcfebd36be37971a5a9e",
    "DataExplanation.md": "b3f76692a68414cf95c3bb00b386060bdcfa9769",
}
STATE_COLUMNS = ["Theta1", "Theta2", "dTheta1", "dTheta2"]


def source_files(directory, download):
    """Verify immutable Git blobs, retaining upstream attribution and license."""
    directory.mkdir(parents=True, exist_ok=True)

    def obtain(item):
        name, expected = item
        path = directory / name
        url = f"https://api.github.com/repos/dynamicslab/MultiArm-Pendulum/git/blobs/{expected}"
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing {path}; pass --download to fetch the pinned source.")
            request = urllib.request.Request(url, headers={"User-Agent": "adaptive-solver-bench-example"})
            with urllib.request.urlopen(request, timeout=45) as response:
                data = json.load(response)
            raw = base64.b64decode(data["content"], validate=False)
        else:
            raw = path.read_bytes()
        actual = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if actual != expected:
            raise ValueError(f"Source checksum mismatch: {name}")
        if not path.exists():
            path.write_bytes(raw)
        return {"file": name, "git_blob_sha1": actual, "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw), "url": url}

    with ThreadPoolExecutor(max_workers=4) as pool:
        entries = list(pool.map(obtain, FILES.items()))
    provenance = {"repository": REPOSITORY, "commit": COMMIT, "files": entries}
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory.")
    provenance = source_files(args.source, args.download)
    datasets = {"train": [], "validation": [], "test": []}
    audit = []
    for index in range(1, 5):
        name = f"DoubleDataFreeSwing_{index}_Dt_0_001.mat"
        data = loadmat(args.source / name, simplify_cells=True)
        times = np.asarray(data["Time"], dtype=float).reshape(-1)
        states = np.column_stack([np.asarray(data[key]).reshape(-1) for key in STATE_COLUMNS])
        # Validate the complete recording before making the prespecified excerpt.
        full = Trajectory(times, states, name)
        if not np.allclose(np.diff(times), .001, rtol=1e-7, atol=1e-10):
            raise ValueError(f"Unexpected sampling interval: {name}")
        use = times <= times[0] + 5.0
        record = Trajectory(times[use] - times[0], states[use], name.removesuffix(".mat"))
        role = "train" if index <= 2 else "validation" if index == 3 else "test"
        datasets[role].append(record)
        audit.append({"file": name, "role": role, "full_samples": len(full.t),
                      "full_duration_seconds": float(times[-1] - times[0]),
                      "pilot_samples": len(record.t), "full_fingerprint": full.fingerprint})
    config = BenchmarkConfig(selection_metric="normalized_rmse", selection_policy="fastest_within_target",
                             accuracy_target=.1, timing_repeats=3)
    result = benchmark_trajectories(**datasets, config=config)
    result.context.update({"data_kind": "experimental double pendulum; fixed cart; free swing",
                           "source": provenance, "state_columns": STATE_COLUMNS, "data_audit": audit,
                           "horizon_seconds": 5, "preprocessing": "None; native 1 ms samples; supplied velocities",
                           "scope": "Data-driven autonomous prediction; no supplied physical RHS or parameter calibration",
                           "limitations": ["One validation and one test recording; no generalization interval.",
                                           "Separate source files are treated as separate experimental runs; subject/device independence is not established.",
                                           "Upstream velocity estimation/filtering can affect apparent forecasting performance.",
                                           "Targets include measurement and model error; these are not numerical truncation errors.",
                                           "The 0.1 training-standard-deviation target is a pilot convention, not an application safety threshold."]})
    result.save(args.output)
    inputs = args.output / "inputs"
    inputs.mkdir()
    manifest = {"time_column": "time", "state_columns": STATE_COLUMNS}
    for role, records in datasets.items():
        manifest[role] = []
        for record in records:
            relative = f"inputs/{record.name}.csv"
            np.savetxt(args.output / relative, np.column_stack([record.t, record.states]), delimiter=",",
                       header=",".join(["time"] + STATE_COLUMNS), comments="")
            manifest[role].append(relative)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "data-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Selected: {result.selected_method or 'no method meets the validation target'}")
    for row in result.ranking:
        print(row["method"], "validation:", row["validation_score"], "test:", row["test_score"], row["test_status"])
    print(f"Report: {(args.output / 'report.md').resolve()}")
    # A completed comparison with no qualifying candidate is a valid outcome.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
