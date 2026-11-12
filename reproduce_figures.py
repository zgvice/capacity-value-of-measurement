"""Draw the six numerical figures from the supplied result tables."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SCRIPTS = (
    "plot_records_manuscript.py",
    "plot_evidence_manuscript.py",
    "plot_duration_manuscript.py",
    "plot_event_manuscript.py",
    "plot_empirical_aggregation.py",
    "plot_risk_coverage.py",
)


if __name__ == "__main__":
    for name in SCRIPTS:
        print(f"Drawing with {name}", flush=True)
        subprocess.run(
            [sys.executable, "-B", "-X", "utf8", str(ROOT / "figures" / name)],
            cwd=ROOT,
            check=True,
        )
