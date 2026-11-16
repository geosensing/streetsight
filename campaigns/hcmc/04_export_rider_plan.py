"""Export HCMC daily routes to missing-women-rider-route-tool plan.json format."""

import json
from pathlib import Path

import pandas as pd

from gsview.routing import haversine_m

CAMPAIGN_DIR = Path(__file__).resolve().parent
ROOT_DIR = CAMPAIGN_DIR.parents[1]

MAX_STOPS_PER_TRIP = 15


def _trip_distance_km(coords: list[list[float]]) -> float:
    """Compute straight-line route distance in km across ordered [lat, lon] stops."""
    if len(coords) < 2:
        return 0.0
    total_m = sum(
        haversine_m(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
        for i in range(len(coords) - 1)
    )
    return round(total_m / 1000.0, 2)


def _google_maps_nav_url(coords: list[list[float]]) -> str:
    """Build Google Maps multi-stop navigation URL starting from Current Location."""
    waypoints = "/".join(f"{lat:.6f},{lon:.6f}" for lat, lon in coords)
    return f"https://www.google.com/maps/dir/Current+Location/{waypoints}"


def build_rider_plan(
    routes: pd.DataFrame, max_stops_per_trip: int = MAX_STOPS_PER_TRIP
) -> dict:
    """Convert HCMC daily route DataFrame into rider-route-tool plan.json structure."""
    days_payload = []

    for day_num, group in routes.groupby("collection_day"):
        day_int = int(day_num)
        ordered = group.sort_values("stop_order").reset_index(drop=True)
        window = ordered["time_window"].iloc[0]
        all_coords = [[float(r["lat"]), float(r["lon"])] for _, r in ordered.iterrows()]

        trips = []
        day_route_km = 0.0
        for part_idx, start in enumerate(
            range(0, len(all_coords), max_stops_per_trip), start=1
        ):
            chunk = all_coords[start : start + max_stops_per_trip]
            trip_km = _trip_distance_km(chunk)
            day_route_km += trip_km
            trips.append(
                {
                    "itineraryId": f"hcmc_d{day_int:02d}",
                    "part": part_idx,
                    "title": f"HCMC Day {day_int:02d} ({window}) - Part {part_idx}",
                    "stopCount": len(chunk),
                    "routeKm": trip_km,
                    "coords": chunk,
                    "navigationUrl": _google_maps_nav_url(chunk),
                    "sequenceInDay": part_idx,
                    "endsAtHome": False,
                }
            )

        days_payload.append(
            {
                "day": day_int,
                "itineraryIds": [f"hcmc_d{day_int:02d}"],
                "tripCount": len(trips),
                "partCount": len(trips),
                "stopCount": len(all_coords),
                "routeKm": round(day_route_km, 2),
                "actualDayKm": round(day_route_km, 2),
                "estimatedCoverageKm": round(day_route_km, 2),
                "trips": trips,
            }
        )

    return {
        "project": "Missing Women Rider Route Tool - HCMC",
        "campaign": "hcmc",
        "city": "Ho Chi Minh City",
        "days": days_payload,
    }


def main() -> None:
    """Generate data/hcmc/routes/plan.json from hcmc_routes_all.csv."""
    cfg = json.loads((CAMPAIGN_DIR / "config.json").read_text(encoding="utf-8"))
    routes_dir = ROOT_DIR / cfg["data_dir"] / "routes"
    routes_path = routes_dir / "hcmc_routes_all.csv"

    if not routes_path.exists():
        raise FileNotFoundError(
            f"Missing {routes_path}. Run 03_build_daily_routes.py first."
        )

    routes = pd.read_csv(routes_path)
    plan = build_rider_plan(routes)

    out_path = routes_dir / "plan.json"
    out_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Exported {len(plan['days'])} days to {out_path}")


if __name__ == "__main__":
    main()
