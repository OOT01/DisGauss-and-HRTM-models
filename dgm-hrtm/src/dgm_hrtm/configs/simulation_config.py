from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SimulationConfig:
    # ----- Domain -----
    max_x: float
    max_y: float
    grid_resolution: int
    evaluation_height_z: float

    # ----- Source -----
    emission_rate_Q: float
    emission_height_H: float
    wind_components: List[float]

    # ----- Dispersion -----
    dispersion_model: str
    stability_index: int
    colormap_index: int

    # ----- Radionuclide / Dose -----
    radionuclide: str
    population: str                  # "worker" or "public"
    exposure_pathway: str            # "inhalation" or "ingestion"
    absorption_type: Optional[str]   # e.g. "F", "M", "S" when applicable
    age_group: Optional[str]         # only for public
    amad: Optional[str]              # e.g. "1μm" or "5μm"
    ingestion_key: Optional[str]

    # ----- HRTM -----
    gender_subject: Optional[str]        # None, "male", "female"
    breathing_mode: Optional[str]        # "nasal" or "oral"
    particle_shape_factor: Optional[float]
    particle_density_g_cm3: Optional[float]

    # ----- Meteorology and Mapbox -----
    coordinates: List[float]         # [lat, lon]
    meteo_mode: str                  # "historical", "current", "forecast"
    meteo_date: Optional[str]
    meteo_hour: Optional[int]

    # ----- Exposure -----
    exposure_time_h: float

    # ----- Optional flags -----
    use_regional_sf: bool = False
    use_mapbox: bool = False