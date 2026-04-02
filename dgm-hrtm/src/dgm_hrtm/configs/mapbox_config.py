from dataclasses import dataclass


@dataclass
class MapboxConfig:
    enabled: bool = False
    username: str = "mapbox"
    style_id: str = "satellite-v9"   # e.g. "satellite-v9", "streets-v12"
    width: int = 1280
    height: int = 1280
    highres: bool = True
    padding: int = 0
    alpha: float = 1.0
    save_background: bool = False
    background_filename: str = "mapbox_background.png"