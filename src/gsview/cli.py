"""Command-line interface for gsview."""

import click
import pandas as pd

from .downloader import (
    check_coverage_batch,
    download_images_batch,
    download_images_hires_batch,
    generate_annotation_csv,
    print_coverage_stats,
)
from .sampling import plot_samples, sample_all_cities, sample_city


@click.group()
def main():
    """Google Street View sampling and download tool."""


@main.command()
@click.option(
    "--city",
    type=click.Choice(["mumbai", "delhi", "navi_mumbai", "hcmc", "all"]),
    default="all",
)
@click.option(
    "-n",
    "--n-samples",
    type=int,
    default=None,
    help="Number of samples (overrides city default)",
)
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("-o", "--output", type=click.Path(), default="data/samples/locations.csv")
def sample(city, n_samples, seed, output):
    """Sample road locations from cities."""
    click.echo(f"Sampling locations (seed={seed})...")

    if city == "all":
        df = sample_all_cities(seed=seed)
    else:
        df = sample_city(city, n_samples=n_samples, seed=seed)
        df["location_id"] = [f"loc_{i:05d}" for i in range(len(df))]
        df = df[["location_id", "city", "lat", "lon"]]

    df.to_csv(output, index=False)
    click.echo(f"Sampled {len(df)} locations")
    click.echo(f"Output: {output}")

    for c in df["city"].unique():
        count = len(df[df["city"] == c])
        click.echo(f"  {c}: {count}")


@main.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(exists=True),
    default="data/samples/locations.csv",
)
@click.option("-o", "--output", type=click.Path(), default="data/coverage/coverage.csv")
@click.option("--rate-limit", type=float, default=0.1, help="Seconds between API calls")
def coverage(input_path, output, rate_limit):
    """Check Street View coverage for sampled locations."""
    locations = pd.read_csv(input_path)
    click.echo(f"Checking coverage for {len(locations)} locations...")

    results = check_coverage_batch(
        locations,
        output_path=output,
        rate_limit=rate_limit,
    )

    print_coverage_stats(results)
    click.echo(f"\nOutput: {output}")


@main.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(exists=True),
    default="data/coverage/coverage.csv",
)
@click.option("-o", "--output-dir", type=click.Path(), default="data/images")
@click.option(
    "--headings", type=str, default="0,90,180,270", help="Comma-separated headings"
)
@click.option("--pitch", type=int, default=0, help="Camera pitch (-90 to 90)")
@click.option("--rate-limit", type=float, default=0.1, help="Seconds between API calls")
@click.option("--skip-existing/--no-skip-existing", default=True)
@click.option(
    "--hires/--no-hires",
    default=False,
    help="Use high-res panorama tiles (no API key needed, ~2048px output)",
)
@click.option(
    "--zoom",
    type=int,
    default=3,
    help="Panorama zoom level for hi-res mode (0-5). 3=~2048px, 4=~4096px",
)
def download(
    input_path, output_dir, headings, pitch, rate_limit, skip_existing, hires, zoom
):
    """Download Street View images for locations with coverage."""
    coverage_df = pd.read_csv(input_path)
    locations = coverage_df[coverage_df["has_coverage"]].copy()
    click.echo(f"Downloading images for {len(locations)} locations with coverage...")

    heading_list = [int(h) for h in headings.split(",")]
    click.echo(f"Headings: {heading_list}, Pitch: {pitch}")

    total_images = len(locations) * len(heading_list)
    click.echo(f"Total images to download: {total_images}")

    if hires:
        click.echo(f"Mode: High-resolution (zoom={zoom})")
        if "pano_id" not in locations.columns:
            click.echo("Error: --hires requires pano_id column in input CSV", err=True)
            click.echo("Run 'gsview coverage' first to get pano_ids.", err=True)
            return

        results = download_images_hires_batch(
            locations,
            output_dir=output_dir,
            headings=heading_list,
            pitch=pitch,
            zoom=zoom,
            skip_existing=skip_existing,
        )
    else:
        click.echo("Mode: Standard (640x640 API)")
        results = download_images_batch(
            locations,
            output_dir=output_dir,
            headings=heading_list,
            pitch=pitch,
            rate_limit=rate_limit,
            skip_existing=skip_existing,
        )

    successful = results["success"].sum()
    click.echo(f"\nDownloaded: {successful}/{len(results)} images")

    results_path = output_dir + "/download_results.csv"
    results.to_csv(results_path, index=False)
    click.echo(f"Results: {results_path}")


@main.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(exists=True),
    default="data/images/download_results.csv",
)
@click.option("-o", "--output", type=click.Path(), default="data/annotation.csv")
def annotate(input_path, output):
    """Generate annotation CSV from download results."""
    results = pd.read_csv(input_path)
    click.echo(f"Processing {len(results)} download results...")

    annotation_df = generate_annotation_csv(results, output)
    click.echo(f"Generated {len(annotation_df)} annotation rows")
    click.echo(f"Output: {output}")


@main.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(exists=True),
    default="data/samples/locations.csv",
)
@click.option("-o", "--output", type=click.Path(), default="data/samples/map.html")
def plot(input_path, output):
    """Create interactive map of sampled locations."""
    locations = pd.read_csv(input_path)
    click.echo(f"Plotting {len(locations)} locations...")

    for city in locations["city"].unique():
        count = len(locations[locations["city"] == city])
        click.echo(f"  {city}: {count}")

    output_path = plot_samples(locations, output)
    click.echo(f"Map saved to: {output_path}")


if __name__ == "__main__":
    main()
