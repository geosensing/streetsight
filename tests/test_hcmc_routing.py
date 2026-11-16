"""Tests for HCMC city configuration and daily route clustering."""

from pathlib import Path

import pandas as pd

from gsview.routing import assign_daily_routes, export_routes_gpx, plot_daily_routes
from gsview.sampling import (
    CITY_CONFIGS,
    _segment_roads,
    attach_edge_lengths,
    sample_city,
)


def test_hcmc_city_config():
    assert "hcmc" in CITY_CONFIGS
    cfg = CITY_CONFIGS["hcmc"]
    assert cfg.name == "Ho Chi Minh City"
    assert cfg.bbox is not None
    assert len(cfg.bbox) == 4


def test_sample_city_hcmc_uses_custom_roads_dir(tmp_path: Path):
    roads_dir = tmp_path / "roads"
    roads_dir.mkdir(parents=True)
    mock_roads = pd.DataFrame(
        {
            "lat": [10.77 + i * 0.001 for i in range(20)],
            "lon": [106.68 + i * 0.001 for i in range(20)],
            "osm_id": list(range(100, 120)),
            "osm_name": [f"Street {i}" for i in range(20)],
            "osm_type": ["primary" if i % 2 == 0 else "residential" for i in range(20)],
            "segment_id": list(range(20)),
        }
    )
    mock_roads.to_csv(roads_dir / "hcmc_roads.csv", index=False)

    sampled = sample_city("hcmc", n_samples=10, seed=42, roads_dir=roads_dir)
    assert len(sampled) == 10
    assert (sampled["city"] == "Ho Chi Minh City").all()
    assert set(sampled.columns) == {
        "lat",
        "lon",
        "city",
        "segment_id",
        "osm_name",
        "osm_type",
    }


def test_daily_route_clustering_and_exports(tmp_path: Path):
    points = pd.DataFrame(
        {
            "location_id": [f"hcmc_{i:05d}" for i in range(30)],
            "lat": [10.70 + (i // 6) * 0.03 + (i % 6) * 0.002 for i in range(30)],
            "lon": [106.60 + (i % 6) * 0.03 for i in range(30)],
            "osm_type": ["residential"] * 30,
            "osm_name": [f"Road {i}" for i in range(30)],
        }
    )

    routes = assign_daily_routes(points, n_days=5, seed=42)
    assert len(routes) == 30
    assert set(routes["collection_day"].unique()) == {1, 2, 3, 4, 5}
    for _, group in routes.groupby("collection_day"):
        assert list(group["stop_order"]) == list(range(1, len(group) + 1))

    gpx_path = export_routes_gpx(routes, tmp_path / "routes.gpx")
    assert gpx_path.exists()
    assert "<trk>" in gpx_path.read_text(encoding="utf-8")

    map_path = plot_daily_routes(routes, tmp_path / "routes.html")
    assert map_path.exists()


def test_attach_edge_lengths_matches_segment_midpoints(monkeypatch):
    # A 100 m edge stays whole; a ~1,650 m edge splits into three ~550 m pieces.
    way = {
        "osm_id": 7,
        "name": "",
        "highway": "residential",
        "coords": [
            (10.0, 106.0),
            (10.0, 106.0 + 100 / 109_550),
            (10.0, 106.0 + 1750 / 109_550),
        ],
    }
    monkeypatch.setattr("gsview.sampling._fetch_ways_by_id", lambda ids: [way])
    pieces = pd.DataFrame(_segment_roads([way]))
    pieces["segment_id"] = range(len(pieces))
    samples = pieces[["lat", "lon", "segment_id"]].copy()
    samples.loc[len(samples)] = [10.5, 106.5, 0]  # off the road: must not match

    out = attach_edge_lengths(samples, pieces[["segment_id", "osm_id"]])
    assert list(out["length_m"].round(-1)[:4]) == [100, 550, 550, 550]
    assert out["length_m"].isna().iloc[4]


def test_route_map_escapes_osm_names(tmp_path: Path):
    points = pd.DataFrame(
        {
            "location_id": ["a", "b"],
            "lat": [10.70, 10.71],
            "lon": [106.60, 106.61],
            "osm_type": ["residential", "residential"],
            "osm_name": ["<img src=x onerror=alert(1)>", "${globalThis.p=1}`\\"],
        }
    )
    routes = assign_daily_routes(points, n_days=1)
    text = plot_daily_routes(routes, tmp_path / "m.html").read_text(encoding="utf-8")
    assert "<img src=x" not in text
    assert "&lt;img src=x" in text
    # Popups sit in a JS template literal: no interpolation, no early close.
    assert "${globalThis" not in text
    assert "&#36;{globalThis.p=1}&#96;&#92;" in text
