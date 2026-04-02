import numpy as np
from datetime import datetime

from dgm_hrtm.configs.mapbox_config import MapboxConfig
from dgm_hrtm.dissgausspy import lib_ModelDissGauss as modelo
from dgm_hrtm.functions import func_utils_meteo as utm
from dgm_hrtm.functions import func_utils_printeos as utp


def main_experiment(config, df_rad, dcf_data, results_dir):
    """
    Runs the full Gaussian dispersion + HRTM workflow for a given simulation
    configuration. The function retrieves meteorological data, updates the
    wind components in the configuration, prints a summary of the selected
    settings, and executes the Gaussian dispersion model.
    """

    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # =========================================================
    # MAPBOX CONFIGURATION
    # =========================================================
    if config.use_mapbox:
        map_cfg = MapboxConfig(
            enabled=True,
            username="mapbox",
            style_id="satellite-v9",
            width=1280,
            height=1280,
            highres=True,
            padding=0,
            alpha=0.55,
            save_background=False,
        )
    else:
        map_cfg = None

    # =========================================================
    # GET METEOROLOGICAL DATA
    # =========================================================
    meteo = utm.get_meteo(
        config.coordinates,
        config.meteo_mode,
        config.meteo_date,
        config.meteo_hour,
    )

    config.wind_components = [meteo["u_x"], meteo["u_y"]]

    # =========================================================
    # PRINT METEOROLOGICAL SUMMARY
    # =========================================================
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    utp.print_section("METEOROLOGICAL DATA SELECTED", BLUE + BOLD, RESET)

    utp.print_param("Time", meteo.get("time"), float_format=".3f")
    utp.print_param("Temperature 2m", meteo.get("temperature_2m"), "°C", ".3f")
    utp.print_param("Relative humidity 2m", meteo.get("relative_humidity_2m"), "%", ".3f")
    utp.print_param("Surface pressure", meteo.get("surface_pressure"), "hPa", ".3f")
    utp.print_param("Precipitation", meteo.get("precipitation"), "mm", ".3f")
    utp.print_param("Cloud cover", meteo.get("cloud_cover"), "%", ".3f")
    utp.print_param("Precipitation probability", meteo.get("precipitation_probability"), "%", ".3f")
    utp.print_param("Wind speed 10m", meteo.get("wind_speed_10m"), "m/s", ".3f")
    utp.print_param("Wind direction 10m", meteo.get("wind_direction_10m"), "deg", ".3f")
    utp.print_param("Wind component u_x", meteo.get("u_x"), "m/s", ".3f")
    utp.print_param("Wind component u_y", meteo.get("u_y"), "m/s", ".3f")

    utp.print_end(BLUE, RESET)

    # =========================================================
    # LOAD DCF DATA FOR SUMMARY
    # =========================================================
    nucl = config.radionuclide
    ruta = config.exposure_pathway
    population = config.population

    YELLOW = "\033[93m"

    utp.print_section("DCF PARAMETERS USED", YELLOW, RESET)

    utp.print_param("Radionuclide", nucl)
    utp.print_param("Population", population)
    utp.print_param("Route", ruta)

    dcf_value = None

    if population == "worker":
        if ruta == "inhalation":
            tipo = config.absorption_type
            amad = config.amad
            key = f"e_{amad}"

            dcf_value = float(dcf_data[nucl]["inhalation"][tipo][key])

            utp.print_param("Absorption type", tipo)
            utp.print_param("AMAD", amad)
            utp.print_param("DCF", dcf_value, "Sv/Bq", ".3e")

        elif ruta == "ingestion":
            ing_key = config.ingestion_key
            dcf_value = float(dcf_data[nucl]["ingestion"][ing_key]["e"])

            utp.print_param("Ingestion key", ing_key)
            utp.print_param("DCF", dcf_value, "Sv/Bq", ".3e")

        else:
            raise ValueError(f"Unknown exposure pathway: {ruta}")

    elif population == "public":
        age_group = config.age_group

        if ruta == "inhalation":
            tipo = config.absorption_type
            dcf_value = float(dcf_data[nucl]["inhalation"][tipo][age_group]["e"])

            utp.print_param("Absorption type", tipo)
            utp.print_param("Age group", age_group)
            utp.print_param("DCF", dcf_value, "Sv/Bq", ".3e")

        elif ruta == "ingestion":
            ing_key = config.ingestion_key
            dcf_value = float(dcf_data[nucl]["ingestion"][ing_key][age_group]["e"])

            utp.print_param("Ingestion key", ing_key)
            utp.print_param("Age group", age_group)
            utp.print_param("DCF", dcf_value, "Sv/Bq", ".3e")

        else:
            raise ValueError(f"Unknown exposure pathway: {ruta}")

    else:
        raise ValueError(f"Unknown population: {population}")

    utp.print_end(YELLOW, RESET)

    # =========================================================
    # PRINT SIMULATION SUMMARY
    # =========================================================
    GREEN = "\033[92m"

    half_life_str = dcf_data[nucl]["half_life"]
    half_life = float(half_life_str[:-1])
    unit = half_life_str[-1]

    if unit == "y":
        half_life_seconds = half_life * 365.25 * 24 * 3600
    elif unit == "d":
        half_life_seconds = half_life * 24 * 3600
    elif unit == "h":
        half_life_seconds = half_life * 3600
    elif unit == "m":
        half_life_seconds = half_life * 60
    elif unit == "s":
        half_life_seconds = half_life
    else:
        raise ValueError(f"Unknown half-life unit: {unit}")

    lambda_ = np.log(2) / half_life_seconds
    u_total = np.sqrt(config.wind_components[0] ** 2 + config.wind_components[1] ** 2)

    utp.print_section("SIMULATION CONFIGURATION", GREEN, RESET)

    utp.print_param("Run timestamp", run_timestamp)
    utp.print_param("Dispersion model", config.dispersion_model)
    utp.print_param("Nuclide", nucl)
    utp.print_param("Population", config.population)
    utp.print_param("Exposure pathway", ruta)
    utp.print_param("Absorption type", config.absorption_type)
    utp.print_param("Age group", config.age_group)

    if config.age_group in ["adult", "age_15y"]:
        utp.print_param("Gender", config.gender_subject)

    utp.print_param("AMAD", config.amad)
    utp.print_param("HRTM exposure time", config.exposure_time_h, "h", ".3f")
    utp.print_param("Mapbox background", config.use_mapbox)

    utp.print_subsection("Radiological parameters")
    utp.print_param("Dose coefficient", dcf_value, "Sv/Bq", ".3e")
    utp.print_param("Half-life", half_life, unit, ".3f")
    utp.print_param("Decay constant λ", lambda_, "1/s", ".6e")
    utp.print_param("Particle shape factor χ", config.particle_shape_factor)
    utp.print_param("Particle density", config.particle_density_g_cm3, "g/cm³", ".3f")

    utp.print_subsection("Meteorological parameters")
    utp.print_param("Coordinates", config.coordinates)
    utp.print_param("Mode", config.meteo_mode)

    if config.meteo_mode == "current":
        utp.print_param("Date", "-")
        utp.print_param("Hour", "-")
    else:
        utp.print_param("Date", config.meteo_date)
        utp.print_param("Hour", config.meteo_hour)

    utp.print_param("Wind velocity components", config.wind_components, "m/s", ".3f")
    utp.print_param("Total wind speed", u_total, "m/s", ".3f")

    utp.print_subsection("Model parameters")
    utp.print_param("Maximum X", config.max_x, "m", ".3f")
    utp.print_param("Maximum Y", config.max_y, "m", ".3f")
    utp.print_param("Grid resolution", config.grid_resolution, "points", ".3f")
    utp.print_param("Evaluation height Z", config.evaluation_height_z, "m", ".3f")
    utp.print_param("Emission rate Q", config.emission_rate_Q, "Bq/s", ".3f")
    utp.print_param("Effective emission height H", config.emission_height_H, "m", ".3f")
    utp.print_param("Stability index", config.stability_index)
    utp.print_param("Colormap index", config.colormap_index)

    utp.print_end(GREEN, RESET)

    # =========================================================
    # RUN MODEL
    # =========================================================
    model_results = modelo.MododeloContaminacionGaussiana(
        config,
        df_rad,
        dcf_data,
        results_dir,
        meteo=meteo,
        map_cfg=map_cfg,
    )

    results = {
        "dataframes": {},
        "arrays": {},
        "figures": {},
        "metadata": {
            "run_timestamp": run_timestamp,
            "results_dir": results_dir,
            "radionuclide": nucl,
            "population": population,
            "exposure_pathway": ruta,
            "dcf_value": dcf_value,
            "half_life": half_life,
            "half_life_unit": unit,
            "decay_constant_lambda": lambda_,
            "u_total": u_total,
            "meteo": meteo,
        },
    }

    if isinstance(model_results, dict):
        results["dataframes"].update(model_results.get("dataframes", {}))
        results["arrays"].update(model_results.get("arrays", {}))
        results["figures"].update(model_results.get("figures", {}))
        results["metadata"].update(model_results.get("metadata", {}))

    return results