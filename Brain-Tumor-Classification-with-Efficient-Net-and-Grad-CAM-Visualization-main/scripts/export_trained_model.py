from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy the trained checkpoint produced by the notebook into the deployment models folder."
    )
    parser.add_argument(
        "source",
        help="Path to the existing trained checkpoint, typically model.h5 from your notebook runtime.",
    )
    parser.add_argument(
        "--destination",
        default="models/model.keras",
        help="Target path for the deployment artifact.",
    )
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    destination = Path(args.destination).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{source}'. Run the notebook training first or download your saved model.h5."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"Copied trained checkpoint to '{destination}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
