import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

from .inputs import InputError
from .pipeline import run


def main():
    parser = argparse.ArgumentParser(description="Rebuild the current CLUTCH rankings from the supplied data pack.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing the six supplied CSV files.")
    parser.add_argument("--output", type=Path, required=True, help="New results folder; existing folders are never overwritten.")
    parser.add_argument("--config-dir", type=Path, default=Path(__file__).resolve().parents[1] / "config")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run(args.data_dir, args.config_dir, args.output)
    except (InputError, FileNotFoundError) as exc:
        logging.error("%s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
