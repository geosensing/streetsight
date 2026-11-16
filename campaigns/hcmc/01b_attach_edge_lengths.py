"""Backfill the road length each sampled HCMC point represents, for weighting."""

import json
from pathlib import Path

import pandas as pd

from gsview.sampling import attach_edge_lengths

CAMPAIGN_DIR = Path(__file__).resolve().parent
ROOT_DIR = CAMPAIGN_DIR.parents[1]


def main() -> None:
    """Add length_m to the GSV and field samples; the drawn rows are unchanged."""
    cfg = json.loads((CAMPAIGN_DIR / "config.json").read_text(encoding="utf-8"))
    data_dir = ROOT_DIR / cfg["data_dir"]
    roads = pd.read_csv(data_dir / "roads" / "hcmc_roads.csv")
    gsv_path = data_dir / "samples" / "hcmc_gsv_sample.csv"
    field_path = data_dir / "samples" / "hcmc_field_sample.csv"

    gsv = pd.read_csv(gsv_path).drop(columns="length_m", errors="ignore")
    gsv = attach_edge_lengths(gsv, roads)
    unmatched = gsv["length_m"].isna().sum()
    if unmatched > 0.01 * len(gsv):
        raise RuntimeError(
            f"{unmatched} of {len(gsv)} points unmatched; OSM drift too large"
        )
    gsv.to_csv(gsv_path, index=False)

    field = pd.read_csv(field_path).drop(columns="length_m", errors="ignore")
    field = field.merge(gsv[["location_id", "length_m"]], on="location_id", how="left")
    field.to_csv(field_path, index=False)

    print(f"Matched {len(gsv) - unmatched}/{len(gsv)} GSV points")
    print(gsv["length_m"].describe().round(1).to_string())


if __name__ == "__main__":
    main()
