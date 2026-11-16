# Ho Chi Minh City (HCMC) 10–15 Day Campaign

This folder keeps the Ho Chi Minh City data collection campaign separate from the frozen Delhi/Mumbai/Navi Mumbai v2 pipeline.

## Design

- **Primary (`streetscope` fieldwork, 10–15 days on the ground):** 300 sampled locations over 12 `collection_day`s, leaving 2–3 buffer days for rain or re-runs.
  - Each day is one spatial cluster (k-means on coordinates), with stops ordered by greedy nearest neighbor (`stop_order`). Clusters are compact but not equal-sized: 10–40 stops per day.
  - Each day gets a randomized time window (`time_window`, from `config.json`), six mornings and six afternoons. Every area is visited once, so a fixed window would make time of day indistinguishable from neighborhood.
  - `collection_day` is the cluster unit in [`compare_with_streetscope`](../../src/gsview/analysis.py). With only 12 unequal clusters, use a small-cluster correction (e.g. wild cluster bootstrap) for intervals.
- **Secondary (`streetsight` GSV benchmark):** 2,500 locations, matching Delhi/Mumbai, checked against free Google Street View metadata. The field sample is a subset of this sample, so the two can be compared location by location.

### Sampling frame and unit

- **Frame:** OSM `primary|secondary|tertiary|residential|unclassified` ways intersecting the box `(10.68, 106.58, 10.88, 106.82)`, the urban core. This is not the administrative city: since the July 2025 merger, HCMC includes Bình Dương and Bà Rịa–Vũng Tàu. Roads crossing the box edge keep their outside parts, so 40 of the 2,500 points lie just outside the box.
- **Unit:** one OSM node-to-node edge (median ~20 m), drawn with equal probability. The same unit is used for Delhi and Mumbai. Edges are not equal lengths, so an unweighted mean is an average over edges, which over-represents densely mapped residential streets per km of road. `length_m` records each point's edge length: weight by it to estimate a road-length average.

## Layout

```text
campaigns/hcmc/
├── config.json                  # seed, sample sizes, n_days, time windows
├── 01_sample_hcmc.py            # OSM roads -> GSV (2,500) and field (300) samples
├── 01b_attach_edge_lengths.py   # backfill length_m into both samples (one-off; rows unchanged)
├── 02_check_gsv_coverage.py     # free GSV metadata check, coverage by road type
├── 03_build_daily_routes.py     # Day 01..12 packets, time windows, GPX, HTML map
└── 04_export_rider_plan.py      # plan.json for missing-women-rider-route-tool

data/hcmc/
├── roads/hcmc_roads.csv         # gitignored OSM cache (downloaded 2026-11-15)
├── samples/                     # frozen samples, committed
├── coverage/hcmc_coverage.csv   # written by 02
└── routes/                      # all-stops CSV, daily CSVs, GPX, map, plan.json
```

## Run order

```bash
.venv/bin/python campaigns/hcmc/01_sample_hcmc.py
.venv/bin/python campaigns/hcmc/01b_attach_edge_lengths.py
.venv/bin/python campaigns/hcmc/02_check_gsv_coverage.py   # needs GOOGLE_STREETVIEW_API_KEY in .env
.venv/bin/python campaigns/hcmc/03_build_daily_routes.py
.venv/bin/python campaigns/hcmc/04_export_rider_plan.py
cp data/hcmc/routes/plan.json ../missing-women-rider-route-tool/hcmc/plan.json
```

Script 02 stops on any API failure (denied key, quota, server error) instead of recording it as "no coverage". The API key must be allowed to call the Street View Static API. If 02 runs after 03, rerun 03 and 04 so the map popups and route files pick up coverage.

To view the route map, serve it over HTTP; browsers block `file://`:

```bash
python3 -m http.server 8765 -d data/hcmc/routes   # then open http://127.0.0.1:8765/hcmc_routes_map.html
```
