"""Exercise HTTP handling, panorama cropping, and batch output offline."""

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import requests
from PIL import Image

from gsview import downloader, streetview


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Tests must not make live requests")

    monkeypatch.setattr(requests.sessions.Session, "request", unexpected)


@pytest.mark.parametrize("status,covered", [("OK", True), ("ZERO_RESULTS", False)])
def test_metadata_status_controls_coverage(monkeypatch, status, covered):
    response = Mock()
    response.json.return_value = {
        "status": status,
        "pano_id": "pano",
        "date": "2026-01",
    }
    get = Mock(return_value=response)
    monkeypatch.setattr(streetview.requests, "get", get)

    result = streetview.StreetViewClient(api_key="test", rate_limit=0).check_coverage(
        19, 73
    )

    assert result.has_coverage is covered
    assert result.status == status
    assert result.pano_id == "pano"
    assert result.capture_date == "2026-01"
    assert get.call_args.kwargs["params"]["location"] == "19,73"
    response.raise_for_status.assert_called_once()


def test_download_rejects_non_image_response(monkeypatch, tmp_path):
    response = Mock(headers={"Content-Type": "text/html"})
    monkeypatch.setattr(streetview.requests, "get", Mock(return_value=response))
    target = tmp_path / "bad.jpg"

    result = streetview.StreetViewClient(api_key="test", rate_limit=0).download_image(
        19, 73, 90, target
    )

    assert not result.success
    assert "Unexpected content type" in result.error
    assert not target.exists()


def test_download_writes_image_bytes(monkeypatch, tmp_path):
    image = BytesIO()
    Image.new("RGB", (8, 8), "red").save(image, format="JPEG")
    response = Mock(headers={"Content-Type": "image/jpeg"}, content=image.getvalue())
    monkeypatch.setattr(streetview.requests, "get", Mock(return_value=response))
    target = tmp_path / "nested" / "good.jpg"

    result = streetview.StreetViewClient(api_key="test", rate_limit=0).download_image(
        19, 73, 90, target
    )

    assert result.success
    assert target.read_bytes() == image.getvalue()


def test_request_failure_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(
        streetview.requests, "get", Mock(side_effect=requests.Timeout("timed out"))
    )
    result = streetview.StreetViewClient(api_key="test", rate_limit=0).download_image(
        19, 73, 90, tmp_path / "missing.jpg"
    )
    assert not result.success
    assert result.error == "timed out"


@pytest.mark.parametrize("heading", [0, 180, 359])
@pytest.mark.parametrize("fov", [90, 91])
def test_panorama_crop_has_exact_width_across_seam(heading, fov):
    image = Image.new("RGB", (360, 180))
    for x in range(360):
        image.paste((x % 256, 10, 20), (x, 0, x + 1, 180))

    cropped = streetview._crop_panorama(image, heading, 0, fov)

    assert cropped.size == (fov, fov)
    left = heading - fov // 2
    assert [cropped.getpixel((x, 0)) for x in range(fov)] == [
        ((left + x) % 360 % 256, 10, 20) for x in range(fov)
    ]


def test_high_resolution_batch_downloads_panorama_once(monkeypatch, tmp_path):
    monkeypatch.setattr(
        streetview.sv,
        "find_panorama_by_id",
        Mock(return_value=SimpleNamespace(lat=19, lon=73)),
    )
    get = Mock(return_value=Image.new("RGB", (360, 180), "blue"))
    monkeypatch.setattr(streetview.sv, "get_panorama", get)

    results = streetview.download_location_hires(
        "pano", "place", tmp_path, headings=[0, 90]
    )

    get.assert_called_once()
    assert len(results) == 2
    assert all(result.success for result in results)
    assert len(list(tmp_path.glob("*.jpg"))) == 2


def test_missing_panorama_id_never_calls_download(monkeypatch, tmp_path):
    download = Mock(side_effect=AssertionError("missing panorama must not download"))
    monkeypatch.setattr(downloader, "download_location_hires", download)
    locations = pd.DataFrame(
        [{"location_id": "a", "city": "Delhi", "lat": 19, "lon": 73, "pano_id": pd.NA}]
    )

    results = downloader.download_images_hires_batch(
        locations, tmp_path, headings=[0, 90]
    )

    assert len(results) == 2
    assert not results["success"].any()
    download.assert_not_called()


def test_annotation_csv_excludes_failed_downloads(tmp_path):
    common = {"city": "Delhi", "lat": 19, "lon": 73, "heading": 0, "pitch": 0}
    downloads = pd.DataFrame(
        [
            {**common, "location_id": "a", "image_path": "a.jpg", "success": True},
            {**common, "location_id": "b", "image_path": None, "success": False},
        ]
    )
    path = tmp_path / "annotations.csv"

    result = downloader.generate_annotation_csv(downloads, path)

    assert result["location_id"].tolist() == ["a"]
    assert result["annotation_id"].tolist() == ["ann_000000"]
    assert pd.read_csv(path)["location_id"].tolist() == ["a"]
