import os
from io import BytesIO

import numpy as np
import requests
from PIL import Image

from dgm_hrtm.configs.mapbox_config import MapboxConfig


def get_mapbox_token(env_var="MAPBOX_TOKEN"):
    """
    Read Mapbox token from an environment variable.
    Returns None if it does not exist.
    """
    token = os.getenv(env_var)
    if token is None or token.strip() == "":
        return None
    return token.strip()


def local_xy_to_latlon(x_m, y_m, lat0_deg, lon0_deg):
    """
    Convert local Cartesian coordinates in meters into latitude/longitude
    using a local tangent-plane approximation.

    x_m : east-west displacement [m]
    y_m : north-south displacement [m]
    """
    lat = lat0_deg + (np.asarray(y_m) / 111320.0)
    lon = lon0_deg + (
        np.asarray(x_m) / (111320.0 * np.cos(np.radians(lat0_deg)))
    )
    return lat, lon


def local_extent_to_bbox(lat0_deg, lon0_deg, x_min, x_max, y_min, y_max):
    """
    Convert a local metric simulation extent into a geographic bounding box:
    [lon_min, lat_min, lon_max, lat_max]
    """
    corners = [
        (x_min, y_min),
        (x_min, y_max),
        (x_max, y_min),
        (x_max, y_max),
    ]

    lats = []
    lons = []

    for x, y in corners:
        lat, lon = local_xy_to_latlon(x, y, lat0_deg, lon0_deg)
        lats.append(float(lat))
        lons.append(float(lon))

    lon_min = min(lons)
    lon_max = max(lons)
    lat_min = min(lats)
    lat_max = max(lats)

    return [lon_min, lat_min, lon_max, lat_max]


def build_mapbox_static_url(
    username,
    style_id,
    bbox,
    width,
    height,
    token,
    highres=True,
    padding=0,
    attribution=False,
    logo=False,
):
    """
    Build a Mapbox Static Images API URL using bbox mode.
    """
    width = int(np.clip(width, 1, 1280))
    height = int(np.clip(height, 1, 1280))

    lon_min, lat_min, lon_max, lat_max = bbox
    bbox_str = f"[{lon_min:.6f},{lat_min:.6f},{lon_max:.6f},{lat_max:.6f}]"
    suffix = "@2x" if highres else ""

    url = (
        f"https://api.mapbox.com/styles/v1/{username}/{style_id}/static/"
        f"{bbox_str}/{width}x{height}{suffix}"
        f"?access_token={token}"
        f"&padding={int(padding)}"
        f"&attribution={'true' if attribution else 'false'}"
        f"&logo={'true' if logo else 'false'}"
    )

    return url


def download_mapbox_image(url, timeout=30):
    """
    Download a Mapbox static image and return it as a NumPy array.
    """
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    img = Image.open(BytesIO(response.content)).convert("RGBA")
    return np.array(img)


def add_mapbox_background(
    ax,
    lat0_deg,
    lon0_deg,
    x_min,
    x_max,
    y_min,
    y_max,
    map_cfg: MapboxConfig | None,
    token=None,
    results_dir=None,
):
    """
    Draw a georeferenced Mapbox background on a matplotlib axis using the same
    metric extent as the Gaussian plume grid.

    Returns True if the map was drawn successfully, False otherwise.
    """
    if map_cfg is None or not map_cfg.enabled:
        return False

    token = token or get_mapbox_token()
    if token is None:
        print("⚠️ Mapbox background disabled: MAPBOX_TOKEN not found.")
        return False

    try:
        bbox = local_extent_to_bbox(
            lat0_deg=lat0_deg,
            lon0_deg=lon0_deg,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

        url = build_mapbox_static_url(
            username=map_cfg.username,
            style_id=map_cfg.style_id,
            bbox=bbox,
            width=map_cfg.width,
            height=map_cfg.height,
            token=token,
            highres=map_cfg.highres,
            padding=map_cfg.padding,
            attribution=False,
            logo=False,
        )

        img = download_mapbox_image(url)

        if map_cfg.save_background and results_dir is not None:
            out_path = os.path.join(results_dir, map_cfg.background_filename)
            Image.fromarray(img).save(out_path)

        ax.imshow(
            img,
            extent=[x_min, x_max, y_min, y_max],
            origin="upper",
            aspect="auto",
            alpha=map_cfg.alpha,
            zorder=0,
        )

        return True

    except Exception as exc:
        print(f"⚠️ Mapbox background could not be loaded: {exc}")
        return False