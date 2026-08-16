from __future__ import annotations

import argparse

from tabulus import __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tabulus",
        description="Scientific PDF table extraction and enrichment pipeline.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.parse_args()


if __name__ == "__main__":
    main()
