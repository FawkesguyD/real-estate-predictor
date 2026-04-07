from __future__ import annotations

from ml.model.main import parse_args, run_pipeline


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        data_path=args.data_path,
        artifacts_dir=args.artifacts_dir,
        reports_dir=args.reports_dir,
        force_download=args.force_download,
    )
