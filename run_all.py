"""
Single runner: generates all the data make_figures.ipynb needs -- data
generation + training + ROC (via pta_flatland.main's pipeline) for every
simple/composite x weak/strong config, then the linear-classifier chance
band (via pta_flatland.chance_band, see its module docstring) for whichever
composite scenarios were run.

Usage:
  python run_all.py                                # the 4 default scenarios
  python run_all.py simple_weak composite_strong    # just these
"""

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "config_files"

SCENARIOS = [
    "simple_weak",
    "simple_strong",
    "composite_weak",
    "composite_strong",
]


def run_one(name: str) -> None:
    config_path = CONFIG_DIR / f"{name}.py"
    if not config_path.exists():
        raise FileNotFoundError(f"No config for scenario {name!r}: {config_path}")
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
    t0 = time.perf_counter()
    subprocess.run(
        [sys.executable, "-m", "pta_flatland.main", "--config", str(config_path)],
        cwd=REPO_ROOT,
        check=True,
    )
    print(f"[{name}] done in {time.perf_counter() - t0:.1f}s", flush=True)


def run_chance_band(name: str) -> None:
    config_path = CONFIG_DIR / f"{name}.py"
    print(f"\n{'=' * 70}\n{name} chance band\n{'=' * 70}", flush=True)
    t0 = time.perf_counter()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pta_flatland.chance_band",
            "--config",
            str(config_path),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    print(f"[{name}] chance band done in {time.perf_counter() - t0:.1f}s", flush=True)


def main() -> None:
    requested = sys.argv[1:] or SCENARIOS
    unknown = [n for n in requested if n not in SCENARIOS]
    if unknown:
        raise SystemExit(f"Unknown scenario(s): {unknown}. Choices: {SCENARIOS}")

    t0 = time.perf_counter()
    for name in requested:
        run_one(name)
    for name in requested:
        if name.startswith("composite_"):
            run_chance_band(name)
    print(
        f"\nAll {len(requested)} scenario(s) done in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
