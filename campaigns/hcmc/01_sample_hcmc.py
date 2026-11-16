"""Sample HCMC OSM road segments for GSV baseline and 10-15 day field collection."""

import json
from pathlib import Path

import pandas as pd

from gsview.sampling import sample_city

CAMPAIGN_DIR = Path(__file__).resolve().parent
ROOT_DIR = CAMPAIGN_DIR.parents[1]


def load_config() -> dict:
    """Load HCMC campaign configuration."""
    return json.loads((CAMPAIGN_DIR / "config.json").read_text(encoding="utf-8"))


def draw_stratified_field_sample(
    gsv_sample: pd.DataFrame,
    n_field: int,
    seed: int,
) -> pd.DataFrame:
    """
    Draw a road-type-stratified field subsample from the GSV sample.

    Allocates `n_field` proportionally across `osm_type` so the field sample
    matches the city's road-network composition and overlaps cleanly with GSV.
    """
    if n_field >= len(gsv_sample):
        return gsv_sample.copy().reset_index(drop=True)

    counts = gsv_sample["osm_type"].value_counts()
    allocations = (counts / counts.sum() * n_field).round().astype(int)

    # Adjust largest stratum if rounding shifts the total away from n_field.
    diff = n_field - int(allocations.sum())
    if diff != 0:
        allocations.iloc[0] += diff

    parts = []
    for i, (road_type, target_n) in enumerate(allocations.items()):
        if target_n <= 0:
            continue
        stratum = gsv_sample[gsv_sample["osm_type"] == road_type]
        parts.append(
            stratum.sample(n=min(target_n, len(stratum)), random_state=seed + i)
        )

    return pd.concat(parts, ignore_index=True).reset_index(drop=True)


def main() -> None:
    """Run HCMC road sampling and save GSV + field sample CSVs."""
    cfg = load_config()
    data_dir = ROOT_DIR / cfg["data_dir"]
    roads_dir = data_dir / "roads"
    samples_dir = data_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    gsv_df = sample_city(
        cfg["city"],
        n_samples=cfg["gsv_sample_n"],
        seed=cfg["seed"],
        roads_dir=roads_dir,
    )
    gsv_df.insert(0, "location_id", [f"hcmc_{i:05d}" for i in range(len(gsv_df))])

    gsv_path = samples_dir / "hcmc_gsv_sample.csv"
    gsv_df.to_csv(gsv_path, index=False)

    field_df = draw_stratified_field_sample(
        gsv_df,
        n_field=cfg["field_sample_n"],
        seed=cfg["seed"],
    )
    field_path = samples_dir / "hcmc_field_sample.csv"
    field_df.to_csv(field_path, index=False)

    print(f"Saved GSV baseline sample ({len(gsv_df)}): {gsv_path}")
    print(f"Saved field sample ({len(field_df)}): {field_path}")
    print("\nField sample by road type:")
    print(field_df["osm_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
