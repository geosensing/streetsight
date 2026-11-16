"""Create Label Studio tasks JSON from downloaded Street View images."""

import json
from pathlib import Path

import pandas as pd


def main():
    data_dir = Path(__file__).parent.parent / "data"

    samples = pd.read_csv(data_dir / "samples" / "random_sample_4500.csv")
    downloads = pd.read_csv(data_dir / "images" / "random_sample.csv")

    merged = downloads.merge(
        samples[["location_id", "segment_id", "osm_name", "osm_type"]],
        on="location_id",
        how="left",
    )

    successful = merged[merged["success"].eq(True)]

    tasks = []
    for _, row in successful.iterrows():
        image_filename = Path(str(row["image_path"])).name
        segment_id = row["segment_id"]
        osm_name = row["osm_name"]
        osm_type = row["osm_type"]
        task = {
            "data": {
                "image": f"gs://sawasdee-labelstudio/google_streetview/{image_filename}",
                "location_id": row["location_id"],
                "city": row["city"],
                "lat": row["lat"],
                "lon": row["lon"],
                "heading": int(row["heading"]),
                "pitch": int(row["pitch"]),
                "segment_id": int(segment_id) if pd.notna(segment_id) else None,
                "osm_name": osm_name if pd.notna(osm_name) else "",
                "osm_type": osm_type if pd.notna(osm_type) else "",
            }
        }
        tasks.append(task)

    output_path = data_dir / "labelstudio_tasks.json"
    with open(output_path, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"Created {len(tasks)} Label Studio tasks")
    print(f"Output: {output_path}")

    cities = pd.DataFrame([t["data"] for t in tasks]).groupby("city").size()
    print("\nTasks by city:")
    for city, count in cities.items():
        print(f"  {city}: {count}")


if __name__ == "__main__":
    main()
