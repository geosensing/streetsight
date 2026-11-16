"""Check free Google Street View coverage for the sampled HCMC locations."""

import json
from pathlib import Path

import pandas as pd

from gsview.downloader import check_coverage_batch, print_coverage_stats

CAMPAIGN_DIR = Path(__file__).resolve().parent
ROOT_DIR = CAMPAIGN_DIR.parents[1]


def main() -> None:
    """Check GSV metadata coverage and summarize by road type and capture year."""
    cfg = json.loads((CAMPAIGN_DIR / "config.json").read_text(encoding="utf-8"))
    data_dir = ROOT_DIR / cfg["data_dir"]

    sample_path = data_dir / "samples" / "hcmc_gsv_sample.csv"
    coverage_path = data_dir / "coverage" / "hcmc_coverage.csv"

    if not sample_path.exists():
        raise FileNotFoundError(f"Missing {sample_path}. Run 01_sample_hcmc.py first.")

    samples = pd.read_csv(sample_path)
    results = check_coverage_batch(samples, output_path=coverage_path, rate_limit=0.1)
    print_coverage_stats(results)

    merged = samples[["location_id", "osm_type"]].merge(
        results, on="location_id", how="left"
    )
    print("\nCoverage rate by OSM road type:")
    by_type = merged.groupby("osm_type")["has_coverage"].agg(["count", "sum", "mean"])
    by_type["mean"] = (by_type["mean"] * 100).round(1)
    print(by_type.rename(columns={"sum": "covered", "mean": "pct_covered"}).to_string())


if __name__ == "__main__":
    main()
