"""Batch operations for Street View coverage checking and image downloading."""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .streetview import StreetViewClient, download_location_hires


def check_coverage_batch(
    locations: pd.DataFrame,
    output_path: str | Path | None = None,
    api_key: str | None = None,
    rate_limit: float = 0.1,
) -> pd.DataFrame:
    """
    Check Street View coverage for a batch of locations.

    Parameters
    ----------
    locations : pd.DataFrame
        DataFrame with columns: location_id, lat, lon, city
    output_path : str or Path, optional
        Path to save results CSV
    api_key : str, optional
        Google API key
    rate_limit : float
        Seconds between API calls

    Returns
    -------
    pd.DataFrame
        Input DataFrame with coverage metadata, returned panorama coordinates, and
        query-to-panorama snap distance.
    """
    client = StreetViewClient(api_key=api_key, rate_limit=rate_limit)
    results = []

    for _, row in tqdm(
        locations.iterrows(), total=len(locations), desc="Checking coverage"
    ):
        result = client.check_coverage(row["lat"], row["lon"])
        results.append(
            {
                "location_id": row["location_id"],
                "city": row["city"],
                "lat": row["lat"],
                "lon": row["lon"],
                "has_coverage": result.has_coverage,
                "pano_id": result.pano_id,
                "capture_date": result.capture_date,
                "pano_lat": result.pano_lat,
                "pano_lon": result.pano_lon,
                "snap_distance_m": result.snap_distance_m,
                "status": result.status,
            }
        )

    df = pd.DataFrame(results)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


def download_images_batch(
    locations: pd.DataFrame,
    output_dir: str | Path,
    headings: list[int] | None = None,
    pitch: int = 0,
    api_key: str | None = None,
    rate_limit: float = 0.1,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """
    Download Street View images for a batch of locations.

    Parameters
    ----------
    locations : pd.DataFrame
        DataFrame with columns: location_id, lat, lon, city.
        Should be filtered to locations with coverage.
    output_dir : str or Path
        Directory to save images
    headings : list[int], optional
        Camera headings. Defaults to [0, 90, 180, 270].
    pitch : int
        Camera pitch
    api_key : str, optional
        Google API key
    rate_limit : float
        Seconds between API calls
    skip_existing : bool
        Skip locations that already have images downloaded

    Returns
    -------
    pd.DataFrame
        Download results with columns: location_id, city, lat, lon, heading, pitch,
        image_path, success, error
    """
    client = StreetViewClient(api_key=api_key, rate_limit=rate_limit)
    headings = headings or [0, 90, 180, 270]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for _, row in tqdm(
        locations.iterrows(), total=len(locations), desc="Downloading images"
    ):
        location_id = row["location_id"]

        if skip_existing:
            expected_files = [
                output_dir / f"{location_id}_h{h:03d}_p{pitch:+03d}.jpg"
                for h in headings
            ]
            if all(f.exists() for f in expected_files):
                for h, f in zip(headings, expected_files):
                    all_results.append(
                        {
                            "location_id": location_id,
                            "city": row["city"],
                            "lat": row["lat"],
                            "lon": row["lon"],
                            "heading": h,
                            "pitch": pitch,
                            "image_path": str(f),
                            "success": True,
                            "error": None,
                        }
                    )
                continue

        pano_id = row.get("pano_id")
        if pd.isna(pano_id):
            pano_id = None
        results = client.download_location_images(
            lat=row["lat"],
            lon=row["lon"],
            location_id=location_id,
            output_dir=output_dir,
            headings=headings,
            pitch=pitch,
            pano_id=pano_id,
        )

        for result in results:
            all_results.append(
                {
                    "location_id": location_id,
                    "city": row["city"],
                    "lat": row["lat"],
                    "lon": row["lon"],
                    "heading": result.heading,
                    "pitch": result.pitch,
                    "image_path": result.image_path,
                    "success": result.success,
                    "error": result.error,
                }
            )

    return pd.DataFrame(all_results)


def download_images_hires_batch(
    locations: pd.DataFrame,
    output_dir: str | Path,
    headings: list[int] | None = None,
    pitch: int = 0,
    fov: int = 90,
    zoom: int = 3,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """
    Download high-resolution Street View images for a batch of locations.

    Uses the streetlevel library to fetch full panoramas and crop them.
    No API key required.

    Parameters
    ----------
    locations : pd.DataFrame
        DataFrame with columns: location_id, lat, lon, city, pano_id.
        Must have pano_id from coverage check.
    output_dir : str or Path
        Directory to save images
    headings : list[int], optional
        Camera headings. Defaults to [0, 90, 180, 270].
    pitch : int
        Camera pitch
    fov : int
        Field of view in degrees
    zoom : int
        Panorama zoom level (0-5). 3 = ~2048px, 4 = ~4096px
    skip_existing : bool
        Skip locations that already have images downloaded

    Returns
    -------
    pd.DataFrame
        Download results with columns: location_id, city, lat, lon, heading, pitch,
        image_path, success, error
    """
    headings = headings or [0, 90, 180, 270]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for _, row in tqdm(
        locations.iterrows(), total=len(locations), desc="Downloading hi-res images"
    ):
        location_id = row["location_id"]
        pano_id = row.get("pano_id")

        if pd.isna(pano_id) or not pano_id:
            for h in headings:
                all_results.append(
                    {
                        "location_id": location_id,
                        "city": row["city"],
                        "lat": row["lat"],
                        "lon": row["lon"],
                        "heading": h,
                        "pitch": pitch,
                        "image_path": None,
                        "success": False,
                        "error": "No pano_id available",
                    }
                )
            continue

        if skip_existing:
            expected_files = [
                output_dir / f"{location_id}_h{h:03d}_p{pitch:+03d}.jpg"
                for h in headings
            ]
            if all(f.exists() for f in expected_files):
                for h, f in zip(headings, expected_files):
                    all_results.append(
                        {
                            "location_id": location_id,
                            "city": row["city"],
                            "lat": row["lat"],
                            "lon": row["lon"],
                            "heading": h,
                            "pitch": pitch,
                            "image_path": str(f),
                            "success": True,
                            "error": None,
                        }
                    )
                continue

        results = download_location_hires(
            pano_id=pano_id,
            location_id=location_id,
            output_dir=output_dir,
            headings=headings,
            pitch=pitch,
            fov=fov,
            zoom=zoom,
        )

        for result in results:
            all_results.append(
                {
                    "location_id": location_id,
                    "city": row["city"],
                    "lat": row["lat"],
                    "lon": row["lon"],
                    "heading": result.heading,
                    "pitch": result.pitch,
                    "image_path": result.image_path,
                    "success": result.success,
                    "error": result.error,
                }
            )

    return pd.DataFrame(all_results)


def generate_annotation_csv(
    download_results: pd.DataFrame,
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Generate annotation CSV for MTurk or similar annotation platform.

    Parameters
    ----------
    download_results : pd.DataFrame
        Results from download_images_batch
    output_path : str or Path
        Path to save annotation CSV

    Returns
    -------
    pd.DataFrame
        Annotation-ready DataFrame
    """
    successful = download_results[download_results["success"]].copy()

    successful["annotation_id"] = [f"ann_{i:06d}" for i in range(len(successful))]

    columns = [
        "annotation_id",
        "location_id",
        "city",
        "lat",
        "lon",
        "heading",
        "pitch",
        "image_path",
    ]

    annotation_df = successful[columns]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_df.to_csv(output_path, index=False)

    return annotation_df


def print_coverage_stats(coverage_df: pd.DataFrame) -> None:
    """Print coverage statistics by city."""
    print("\nCoverage Statistics:")
    print("-" * 50)

    for city in coverage_df["city"].unique():
        city_data = coverage_df[coverage_df["city"] == city]
        total = len(city_data)
        with_coverage = city_data["has_coverage"].sum()
        pct = (with_coverage / total) * 100 if total > 0 else 0
        print(f"{city}: {with_coverage}/{total} ({pct:.1f}%)")

    total = len(coverage_df)
    total_coverage = coverage_df["has_coverage"].sum()
    total_pct = (total_coverage / total) * 100 if total > 0 else 0
    print("-" * 50)
    print(f"Total: {total_coverage}/{total} ({total_pct:.1f}%)")
