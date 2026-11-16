"""Split the HCMC field sample into daily route packets, GPX, and a map."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gsview.routing import assign_daily_routes, export_routes_gpx, plot_daily_routes

CAMPAIGN_DIR = Path(__file__).resolve().parent
ROOT_DIR = CAMPAIGN_DIR.parents[1]


def main() -> None:
    """Build Day 01..Day N route packets for HCMC fieldwork."""
    cfg = json.loads((CAMPAIGN_DIR / "config.json").read_text(encoding="utf-8"))
    data_dir = ROOT_DIR / cfg["data_dir"]

    field_path = data_dir / "samples" / "hcmc_field_sample.csv"
    coverage_path = data_dir / "coverage" / "hcmc_coverage.csv"
    routes_dir = data_dir / "routes"
    daily_dir = routes_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    if not field_path.exists():
        raise FileNotFoundError(f"Missing {field_path}. Run 01_sample_hcmc.py first.")

    field_df = pd.read_csv(field_path)
    if coverage_path.exists():
        cov = pd.read_csv(coverage_path)[
            ["location_id", "has_coverage", "capture_date"]
        ]
        field_df = field_df.merge(cov, on="location_id", how="left")

    routes = assign_daily_routes(field_df, n_days=cfg["n_days"], seed=cfg["seed"])

    # Each area is visited once, so a fixed window would make time of day
    # indistinguishable from neighborhood. Randomize, balanced across days.
    windows = cfg["time_windows"]
    rng = np.random.default_rng(cfg["seed"])
    day_windows = rng.permutation(np.resize(windows, cfg["n_days"]))
    routes["time_window"] = routes["collection_day"].map(
        dict(zip(range(1, cfg["n_days"] + 1), day_windows))
    )

    all_path = routes_dir / "hcmc_routes_all.csv"
    routes.to_csv(all_path, index=False)

    for day, group in routes.groupby("collection_day"):
        day_path = daily_dir / f"day_{int(day):02d}.csv"
        group.to_csv(day_path, index=False)

    gpx_path = export_routes_gpx(routes, routes_dir / "hcmc_routes.gpx")
    map_path = plot_daily_routes(routes, routes_dir / "hcmc_routes_map.html")

    print(f"Saved {len(routes)} stops across {cfg['n_days']} days to {all_path}")
    print(f"Daily CSVs: {daily_dir}")
    print(f"GPX file: {gpx_path}")
    print(f"Interactive map: {map_path}")

    summary = routes.groupby("collection_day").agg(
        window=("time_window", "first"),
        stops=("location_id", "count"),
        route_km=("step_distance_m", lambda s: round(float(s.sum()) / 1000.0, 2)),
    )
    print("\nDaily route summary:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
