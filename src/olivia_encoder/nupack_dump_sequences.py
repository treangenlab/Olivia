"""Step 3 of dataset preparation: list the unique primers NUPACK must fold.

    python -m olivia_encoder.nupack_dump_sequences --embedded-dir <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .data import bundle_sequences, load_bundles

DEFAULT_EMBEDDED_DIR = Path(__file__).resolve().parents[2] / "data" / "embedded" / "true_data_all_t37"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump the unique primer sequences a NUPACK cache must cover.")
    p.add_argument("--embedded-dir", type=Path, default=DEFAULT_EMBEDDED_DIR)
    p.add_argument("--out", type=Path, default=None,
                   help="Destination .txt (default: <embedded-dir>/nupack/sequences.txt).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or (args.embedded_dir / "nupack" / "sequences.txt")
    out.parent.mkdir(parents=True, exist_ok=True)

    bundles = load_bundles(args.embedded_dir)
    seen: dict[str, None] = {}
    for split in ("train", "val", "test"):
        rows = bundle_sequences(bundles[split])
        for s in rows:
            if s:
                seen.setdefault(s, None)
        print(f"  {split}: {len(rows)} rows")

    out.write_text("\n".join(seen) + "\n", encoding="utf-8")
    print(f"\nwrote {len(seen)} unique sequences -> {out}")
    print("next:  python -m olivia_encoder.nupack_targets --embedded-dir %s" % args.embedded_dir)


if __name__ == "__main__":
    main()
