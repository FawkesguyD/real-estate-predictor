from __future__ import annotations

import argparse
import logging

from services.data_migrator.bootstrap import bootstrap_database
from services.data_migrator.importer import resolve_csv_path


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def main() -> int:
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Initialize PostgreSQL schema and import listings from office-sale.csv.",
    )
    parser.add_argument(
        "--csv-path",
        dest="csv_path",
        default=None,
        help="Override the CSV source path. Defaults to ./office-sale.csv or CSV_SOURCE_PATH.",
    )
    args = parser.parse_args()

    resolved_path = resolve_csv_path(args.csv_path)
    stats = bootstrap_database(resolved_path)

    print(f"CSV source: {resolved_path}")
    print(f"Processed rows: {stats.processed}")
    print(f"Imported rows: {stats.imported}")
    print(f"Skipped rows: {stats.skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
