"""Daily route clustering and field-navigation export for multi-day campaigns."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

DAY_PALETTE = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in meters between two WGS84 coordinates."""
    radius_m = 6_371_000.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    return 2.0 * radius_m * atan2(sqrt(a), sqrt(1.0 - a))


def _order_nearest_neighbor(points: pd.DataFrame) -> list[int]:
    """Order row indices of a single day's points via greedy nearest neighbor."""
    if len(points) <= 1:
        return list(points.index)

    remaining = list(points.index)
    # Start at the northernmost point so each day has a deterministic anchor.
    current = points["lat"].idxmax()
    ordered = [current]
    remaining.remove(current)

    while remaining:
        clat = float(points.loc[current, "lat"])
        clon = float(points.loc[current, "lon"])
        nxt = min(
            remaining,
            key=lambda idx: haversine_m(
                clat,
                clon,
                float(points.loc[idx, "lat"]),
                float(points.loc[idx, "lon"]),
            ),
        )
        ordered.append(nxt)
        remaining.remove(nxt)
        current = nxt

    return ordered


def assign_daily_routes(
    locations: pd.DataFrame,
    n_days: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Partition sampled locations into compact spatial clusters for `n_days` of fieldwork.

    Adds `collection_day` (1..n_days), `stop_order` (1..k within day), and
    `step_distance_m` (distance from previous stop on the same day).
    """
    if locations.empty:
        raise ValueError("locations DataFrame is empty")
    if n_days < 1 or n_days > len(locations):
        raise ValueError(f"n_days must be between 1 and {len(locations)}, got {n_days}")

    df = locations.copy().reset_index(drop=True)
    coords = df[["lat", "lon"]].to_numpy(dtype=float)

    # Scale longitude by cos(mean_lat) so k-means distances are approximately isotropic.
    mean_lat_rad = radians(float(coords[:, 0].mean()))
    scaled = np.column_stack([coords[:, 0], coords[:, 1] * cos(mean_lat_rad)])

    _, labels = kmeans2(
        scaled, k=n_days, minit="points", iter=30, rng=np.random.default_rng(seed)
    )

    # Order cluster IDs west-to-east / north-to-south for intuitive day numbering.
    cluster_means = (
        pd.DataFrame({"cluster": labels, "lat": df["lat"], "lon": df["lon"]})
        .groupby("cluster")[["lat", "lon"]]
        .mean()
        .sort_values(["lon", "lat"])
    )
    remap = {
        old_id: new_day for new_day, old_id in enumerate(cluster_means.index, start=1)
    }
    df["collection_day"] = [remap[int(c)] for c in labels]

    df["stop_order"] = 0
    df["step_distance_m"] = 0.0

    for day, group in df.groupby("collection_day"):
        ordered_idx = _order_nearest_neighbor(group)
        for order_num, idx in enumerate(ordered_idx, start=1):
            df.loc[idx, "stop_order"] = order_num
            if order_num > 1:
                prev_idx = ordered_idx[order_num - 2]
                df.loc[idx, "step_distance_m"] = round(
                    haversine_m(
                        float(df.loc[prev_idx, "lat"]),
                        float(df.loc[prev_idx, "lon"]),
                        float(df.loc[idx, "lat"]),
                        float(df.loc[idx, "lon"]),
                    ),
                    1,
                )

    return df.sort_values(["collection_day", "stop_order"]).reset_index(drop=True)


def export_routes_gpx(routes: pd.DataFrame, output_path: str | Path) -> Path:
    """Export daily routes as a GPX 1.1 file with waypoints and per-day tracks."""
    gpx = ET.Element(
        "gpx",
        version="1.1",
        creator="streetsight",
        xmlns="http://www.topografix.com/GPX/1/1",
    )

    for _, row in routes.iterrows():
        wpt = ET.SubElement(
            gpx,
            "wpt",
            lat=f"{float(row['lat']):.6f}",
            lon=f"{float(row['lon']):.6f}",
        )
        name = ET.SubElement(wpt, "name")
        loc_id = row.get("location_id", f"stop_{int(row['stop_order']):02d}")
        name.text = (
            f"D{int(row['collection_day']):02d}-{int(row['stop_order']):02d} {loc_id}"
        )
        desc = ET.SubElement(wpt, "desc")
        osm_type = row.get("osm_type", "")
        osm_name = row.get("osm_name", "")
        desc.text = f"{osm_type} | {osm_name}".strip(" |")

    for day, group in routes.groupby("collection_day"):
        trk = ET.SubElement(gpx, "trk")
        trk_name = ET.SubElement(trk, "name")
        trk_name.text = f"Day {int(day):02d}"
        trkseg = ET.SubElement(trk, "trkseg")
        for _, row in group.sort_values("stop_order").iterrows():
            ET.SubElement(
                trkseg,
                "trkpt",
                lat=f"{float(row['lat']):.6f}",
                lon=f"{float(row['lon']):.6f}",
            )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(gpx)
    ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def plot_daily_routes(routes: pd.DataFrame, output_path: str | Path) -> Path:
    """Create an interactive Folium map showing each collection day as a layer."""
    m = folium.Map(
        location=[float(routes["lat"].mean()), float(routes["lon"].mean())],
        zoom_start=12,
    )

    for day, group in routes.groupby("collection_day"):
        ordered = group.sort_values("stop_order")
        color = DAY_PALETTE[(int(day) - 1) % len(DAY_PALETTE)]
        fg = folium.FeatureGroup(name=f"Day {int(day):02d} ({len(ordered)} stops)")

        coords = [[float(r["lat"]), float(r["lon"])] for _, r in ordered.iterrows()]
        if len(coords) >= 2:
            folium.PolyLine(coords, color=color, weight=2.5, opacity=0.7).add_to(fg)

        for _, row in ordered.iterrows():
            loc_id = row.get("location_id", "")
            osm_type = row.get("osm_type", "")
            osm_name = row.get("osm_name", "")
            gsv = row.get("has_coverage", None)
            gsv_str = f" | GSV={gsv}" if pd.notna(gsv) else ""
            popup = (
                f"Day {int(day):02d} Stop {int(row['stop_order'])} | "
                f"{loc_id} | {osm_type} {osm_name}{gsv_str}"
            )
            folium.CircleMarker(
                location=[float(row["lat"]), float(row["lon"])],
                radius=4,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.85,
                popup=popup,
            ).add_to(fg)

        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    return out
