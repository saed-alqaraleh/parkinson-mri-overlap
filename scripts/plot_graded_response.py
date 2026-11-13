"""Generate Figure 2 using observed (realized) overlap percentages."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.reproduce_figures import figure2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/graded_response.png"))
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure2(args.data_dir, args.out.parent)
    generated = args.out.parent / "Figure_2_graded_response.png"
    if generated != args.out:
        generated.replace(args.out)
    print(f"Reproduced realized-overlap plot: {args.out}")


if __name__ == "__main__":
    main()
