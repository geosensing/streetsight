"""Google Street View API client."""

import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image
from streetlevel import streetview as sv

load_dotenv()

METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"


@dataclass
class CoverageResult:
    """Result of a coverage check."""

    lat: float
    lon: float
    has_coverage: bool
    pano_id: str | None = None
    capture_date: str | None = None
    status: str = "OK"


@dataclass
class ImageResult:
    """Result of an image download."""

    lat: float
    lon: float
    heading: int
    pitch: int
    success: bool
    image_path: str | None = None
    error: str | None = None


class StreetViewClient:
    """Client for Google Street View API."""

    def __init__(
        self,
        api_key: str | None = None,
        rate_limit: float = 0.1,
    ):
        """
        Initialize the client.

        Parameters
        ----------
        api_key : str, optional
            Google API key. If not provided, reads from
            GOOGLE_STREETVIEW_API_KEY env var.
        rate_limit : float
            Minimum seconds between API calls.
        """
        self.api_key = api_key or os.getenv("GOOGLE_STREETVIEW_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set GOOGLE_STREETVIEW_API_KEY env var "
                "or pass api_key."
            )
        self.rate_limit = rate_limit
        self._last_request_time = 0.0

    def _wait_for_rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def check_coverage(self, lat: float, lon: float) -> CoverageResult:
        """
        Check if Street View coverage exists at a location.

        This uses the metadata API which is FREE.

        Parameters
        ----------
        lat : float
            Latitude
        lon : float
            Longitude

        Returns
        -------
        CoverageResult
            Coverage information including pano_id and capture_date if available.
        """
        self._wait_for_rate_limit()

        params = {
            "location": f"{lat},{lon}",
            "key": self.api_key,
        }

        response = requests.get(METADATA_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        status = data.get("status", "UNKNOWN")
        has_coverage = status == "OK"

        return CoverageResult(
            lat=lat,
            lon=lon,
            has_coverage=has_coverage,
            pano_id=data.get("pano_id"),
            capture_date=data.get("date"),
            status=status,
        )

    def download_image(
        self,
        lat: float,
        lon: float,
        heading: int,
        output_path: str | Path,
        pitch: int = 0,
        fov: int = 90,
        size: str = "640x640",
    ) -> ImageResult:
        """
        Download a Street View image.

        This uses the Static API which is PAID (~$7 per 1000 images).

        Parameters
        ----------
        lat : float
            Latitude
        lon : float
            Longitude
        heading : int
            Camera heading (0-360 degrees, 0=North, 90=East)
        output_path : str or Path
            Path to save the image
        pitch : int
            Camera pitch (-90 to 90, negative=down)
        fov : int
            Field of view (10-120 degrees)
        size : str
            Image size (max 640x640 for free tier)

        Returns
        -------
        ImageResult
            Download result with success status.
        """
        self._wait_for_rate_limit()

        params = {
            "location": f"{lat},{lon}",
            "heading": heading,
            "pitch": pitch,
            "fov": fov,
            "size": size,
            "key": self.api_key,
        }

        try:
            response = requests.get(IMAGE_URL, params=params, timeout=60)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type:
                return ImageResult(
                    lat=lat,
                    lon=lon,
                    heading=heading,
                    pitch=pitch,
                    success=False,
                    error=f"Unexpected content type: {content_type}",
                )

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as f:
                f.write(response.content)

            return ImageResult(
                lat=lat,
                lon=lon,
                heading=heading,
                pitch=pitch,
                success=True,
                image_path=str(output_path),
            )

        except requests.RequestException as e:
            return ImageResult(
                lat=lat,
                lon=lon,
                heading=heading,
                pitch=pitch,
                success=False,
                error=str(e),
            )

    def download_location_images(
        self,
        lat: float,
        lon: float,
        location_id: str,
        output_dir: str | Path,
        headings: list[int] | None = None,
        pitch: int = 0,
    ) -> list[ImageResult]:
        """
        Download images for a location at multiple headings.

        Parameters
        ----------
        lat : float
            Latitude
        lon : float
            Longitude
        location_id : str
            Unique location identifier for filenames
        output_dir : str or Path
            Directory to save images
        headings : list[int], optional
            List of headings. Defaults to [0, 90, 180, 270].
        pitch : int
            Camera pitch for all images.

        Returns
        -------
        list[ImageResult]
            Results for each heading.
        """
        headings = headings or [0, 90, 180, 270]
        output_dir = Path(output_dir)
        results = []

        for heading in headings:
            filename = f"{location_id}_h{heading:03d}_p{pitch:+03d}.jpg"
            output_path = output_dir / filename
            result = self.download_image(
                lat=lat,
                lon=lon,
                heading=heading,
                output_path=output_path,
                pitch=pitch,
            )
            results.append(result)

        return results


def download_panorama_hires(
    pano_id: str,
    heading: int,
    output_path: str | Path,
    pitch: int = 0,
    fov: int = 90,
    zoom: int = 3,
) -> ImageResult:
    """
    Download a high-resolution Street View image using streetlevel library.

    This fetches the full panorama and crops it to the specified heading/pitch/fov.
    No API key required (uses internal Google APIs).

    Parameters
    ----------
    pano_id : str
        Panorama ID from coverage check
    heading : int
        Camera heading (0-360 degrees, 0=North, 90=East)
    output_path : str or Path
        Path to save the image
    pitch : int
        Camera pitch (-90 to 90, negative=down)
    fov : int
        Field of view (10-120 degrees)
    zoom : int
        Panorama zoom level (0-5). Higher = more detail.
        3 = ~2048px output, 4 = ~4096px output

    Returns
    -------
    ImageResult
        Download result with success status.
    """
    output_path = Path(output_path)

    try:
        pano = sv.find_panorama_by_id(pano_id)
        if pano is None:
            return ImageResult(
                lat=0.0,
                lon=0.0,
                heading=heading,
                pitch=pitch,
                success=False,
                error=f"Panorama not found: {pano_id}",
            )

        pano_img = sv.get_panorama(pano, zoom=zoom)
        if pano_img is None:
            return ImageResult(
                lat=pano.lat,
                lon=pano.lon,
                heading=heading,
                pitch=pitch,
                success=False,
                error="Failed to download panorama",
            )

        cropped = _crop_panorama(pano_img, heading, pitch, fov)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path, "JPEG", quality=95)

        return ImageResult(
            lat=pano.lat,
            lon=pano.lon,
            heading=heading,
            pitch=pitch,
            success=True,
            image_path=str(output_path),
        )

    except Exception as e:
        return ImageResult(
            lat=0.0,
            lon=0.0,
            heading=heading,
            pitch=pitch,
            success=False,
            error=str(e),
        )


def _crop_panorama(
    img: Image.Image,
    heading: int,
    pitch: int,
    fov: int,
) -> Image.Image:
    """
    Crop an equirectangular panorama to a specific heading/pitch/fov.

    Parameters
    ----------
    img : PIL.Image
        Full equirectangular panorama
    heading : int
        Camera heading (0-360 degrees, 0=center of image for Google panoramas)
    pitch : int
        Camera pitch (-90 to 90)
    fov : int
        Field of view in degrees

    Returns
    -------
    PIL.Image
        Cropped image
    """
    w, h = img.size

    center_x = int((heading / 360.0) * w) % w

    center_y = int(h / 2 - (pitch / 180.0) * h)

    crop_w = int((fov / 360.0) * w)
    crop_h = crop_w

    left = center_x - crop_w // 2
    right = left + crop_w
    raw_top = center_y - crop_h // 2
    top = max(0, raw_top)
    bottom = min(h, raw_top + crop_h)

    if left < 0 or right > w:
        if left < 0:
            left_part = img.crop((w + left, top, w, bottom))
            right_part = img.crop((0, top, right, bottom))
        else:
            left_part = img.crop((left, top, w, bottom))
            right_part = img.crop((0, top, right - w, bottom))

        result = Image.new("RGB", (crop_w, bottom - top))
        result.paste(left_part, (0, 0))
        result.paste(right_part, (left_part.width, 0))
        return result

    return img.crop((left, top, right, bottom))


def download_location_hires(
    pano_id: str,
    location_id: str,
    output_dir: str | Path,
    headings: list[int] | None = None,
    pitch: int = 0,
    fov: int = 90,
    zoom: int = 3,
) -> list[ImageResult]:
    """
    Download high-resolution images for a location at multiple headings.

    Parameters
    ----------
    pano_id : str
        Panorama ID from coverage check
    location_id : str
        Unique location identifier for filenames
    output_dir : str or Path
        Directory to save images
    headings : list[int], optional
        List of headings. Defaults to [0, 90, 180, 270].
    pitch : int
        Camera pitch for all images
    fov : int
        Field of view in degrees
    zoom : int
        Panorama zoom level (0-5)

    Returns
    -------
    list[ImageResult]
        Results for each heading.
    """
    headings = headings or [0, 90, 180, 270]
    output_dir = Path(output_dir)
    results = []

    pano = None
    pano_img = None

    try:
        pano = sv.find_panorama_by_id(pano_id)
        if pano is None:
            for heading in headings:
                results.append(
                    ImageResult(
                        lat=0.0,
                        lon=0.0,
                        heading=heading,
                        pitch=pitch,
                        success=False,
                        error=f"Panorama not found: {pano_id}",
                    )
                )
            return results

        pano_img = sv.get_panorama(pano, zoom=zoom)
        if pano_img is None:
            for heading in headings:
                results.append(
                    ImageResult(
                        lat=pano.lat,
                        lon=pano.lon,
                        heading=heading,
                        pitch=pitch,
                        success=False,
                        error="Failed to download panorama",
                    )
                )
            return results

    except Exception as e:
        for heading in headings:
            results.append(
                ImageResult(
                    lat=0.0,
                    lon=0.0,
                    heading=heading,
                    pitch=pitch,
                    success=False,
                    error=str(e),
                )
            )
        return results

    for heading in headings:
        filename = f"{location_id}_h{heading:03d}_p{pitch:+03d}.jpg"
        output_path = output_dir / filename

        try:
            cropped = _crop_panorama(pano_img, heading, pitch, fov)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(output_path, "JPEG", quality=95)

            results.append(
                ImageResult(
                    lat=pano.lat,
                    lon=pano.lon,
                    heading=heading,
                    pitch=pitch,
                    success=True,
                    image_path=str(output_path),
                )
            )
        except Exception as e:
            results.append(
                ImageResult(
                    lat=pano.lat,
                    lon=pano.lon,
                    heading=heading,
                    pitch=pitch,
                    success=False,
                    error=str(e),
                )
            )

    return results
