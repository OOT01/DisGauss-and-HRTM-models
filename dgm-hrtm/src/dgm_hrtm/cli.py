from dgm_hrtm.configs.simulation_config import SimulationConfig
from dgm_hrtm.runner import load_dcf_data, run_simulation


def _build_prompt(prompt, default=None, allowed=None):
    parts = [prompt]

    if allowed is not None:
        parts.append(f"({', '.join(allowed)})")

    if default is not None:
        parts.append(f"[{default}]")

    return " ".join(parts) + ": "


def _ask_str(prompt, default=None, allowed=None, allow_empty=False):
    while True:
        raw = input(_build_prompt(prompt, default=default, allowed=allowed)).strip()

        if raw == "":
            if default is not None:
                raw = str(default)
            elif allow_empty:
                return None
            else:
                print("This field is required.")
                continue

        if allowed is not None and raw not in allowed:
            print(f"Invalid value. Allowed values: {', '.join(allowed)}")
            continue

        return raw


def _ask_float(prompt, default=None, positive=False):
    while True:
        raw = input(_build_prompt(prompt, default=default)).strip()

        if raw == "":
            if default is not None:
                return float(default)
            print("This field is required.")
            continue

        try:
            value = float(raw)
            if positive and value <= 0:
                print("Value must be positive.")
                continue
            return value
        except ValueError:
            print("Please enter a valid number.")


def _ask_int(prompt, default=None, min_value=None, max_value=None):
    while True:
        raw = input(_build_prompt(prompt, default=default)).strip()

        if raw == "":
            if default is not None:
                return int(default)
            print("This field is required.")
            continue

        try:
            value = int(raw)

            if min_value is not None and value < min_value:
                print(f"Value must be >= {min_value}.")
                continue
            if max_value is not None and value > max_value:
                print(f"Value must be <= {max_value}.")
                continue

            return value
        except ValueError:
            print("Please enter a valid integer.")


def _ask_bool(prompt, default=False):
    default_str = "y" if default else "n"

    while True:
        raw = input(f"{prompt} (y/n) [default={default_str}]: ").strip().lower()

        if raw == "":
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False

        print("Please answer y or n.")


def _ask_valid_radionuclide(population):
    dcf_data = load_dcf_data(population)

    while True:
        radionuclide = _ask_str("Radionuclide")

        if radionuclide not in dcf_data:
            print(
                f"Radionuclide '{radionuclide}' was not found in the {population} DCF database. "
                "Please enter a valid radionuclide."
            )
            continue

        return radionuclide, dcf_data


def _ask_valid_absorption_type(radionuclide, dcf_data):
    while True:
        absorption_type = _ask_str(
            "Absorption type",
            allowed=["F", "M", "S"],
        )

        nucl_data = dcf_data.get(radionuclide, {})
        inhalation_data = nucl_data.get("inhalation", {})

        if absorption_type not in inhalation_data:
            available = list(inhalation_data.keys())

            if available:
                print(
                    f"Absorption type '{absorption_type}' is not available for radionuclide "
                    f"'{radionuclide}'. Available types for this radionuclide: {', '.join(available)}"
                )
            else:
                print(
                    f"Radionuclide '{radionuclide}' does not contain inhalation absorption types "
                    "in the loaded DCF database."
                )
            continue

        return absorption_type


def build_config_interactively():
    print("\n==========================================")
    print(" Insert the required data for the run")
    print("==========================================\n")

    # -------------------------------------------------
    # Domain
    # -------------------------------------------------
    print("DOMAIN PARAMETERS")
    max_x = _ask_float("Maximum X (m)", default=1000.0, positive=True)
    max_y = _ask_float("Maximum Y (m)", default=1000.0, positive=True)
    grid_resolution = _ask_int("Grid resolution", default=200, min_value=10)
    evaluation_height_z = _ask_float("Evaluation height Z (m)", default=2.0, positive=True)

    # -------------------------------------------------
    # Source
    # -------------------------------------------------
    print("\nSOURCE PARAMETERS")
    emission_rate_Q = _ask_float("Emission rate Q (Bq/s)", default=1.0e6, positive=True)
    emission_height_H = _ask_float("Effective emission height H (m)", default=10.0, positive=True)

    # This gets overwritten later from meteo, but the config needs it
    wind_components = [0.0, 0.0]

    # -------------------------------------------------
    # Dispersion
    # -------------------------------------------------
    print("\nDISPERSION PARAMETERS")
    dispersion_model = _ask_str(
        "Dispersion model",
        allowed=["Briggs_r", "Briggs_u", "Pasquill-Gifford"],
    )
    stability_index = _ask_int(
        "Stability index (0 Very unstable, 1 Unstable, 2 Lightly unstable, 3 Neutral, 4 Lightly stable, 5 Stable)",
        min_value=0,
        max_value=5,
    )
    colormap_index = _ask_int("Colormap index (0-20)", min_value=0, max_value=20)

    # -------------------------------------------------
    # Radionuclide / dose
    # -------------------------------------------------
    print("\nRADIONUCLIDE AND DOSE PARAMETERS")
    population = _ask_str(
        "Population",
        allowed=["worker", "public"],
    )

    radionuclide, dcf_data = _ask_valid_radionuclide(population)

    exposure_pathway = _ask_str(
        "Exposure pathway",
        allowed=["inhalation", "ingestion"],
    )

    absorption_type = None
    age_group = None
    amad = None
    ingestion_key = None

    if population == "public":
        age_group = _ask_str(
            "Age group",
            allowed=["infant", "age_1y", "age_5y", "age_10y", "age_15y", "adult"],
        )

    if exposure_pathway == "inhalation":
        absorption_type = _ask_valid_absorption_type(radionuclide, dcf_data)

        amad = _ask_str(
            "AMAD (in μm, but write as '1um' or '5um')",
            allowed=["1um", "5um"],
        )

        if population == "worker":
            print(
                "\nNote: HRTM is not implemented for worker mode in this version. "
                "The simulation will continue with the Gaussian dispersion model only.\n"
            )

    else:
        ingestion_key = _ask_str("Ingestion key", default="f1_0.05")

    # -------------------------------------------------
    # HRTM
    # -------------------------------------------------
    print("\nHRTM PARAMETERS")
    gender_subject = None
    if population == "worker":
        gender_subject = _ask_str(
            "Gender subject",
            allowed=["male", "female"],
        )
    elif age_group in ["adult", "age_15y"]:
        gender_subject = _ask_str(
            "Gender subject",
            allowed=["male", "female"],
        )

    breathing_mode = _ask_str(
        "Breathing mode",
        allowed=["nasal", "oral"],
    )
    particle_shape_factor = _ask_float("Particle shape factor", default=1.5, positive=True)
    particle_density_g_cm3 = _ask_float("Particle density (g/cm^3)", default=3.0, positive=True)
    exposure_time_h = _ask_float("HRTM exposure time (h)", default=1.0, positive=True)
    use_regional_sf = _ask_bool("Use regional sf modifiers?", default=False)

    # -------------------------------------------------
    # Meteorology / coordinates
    # -------------------------------------------------
    print("\nMETEOROLOGICAL PARAMETERS")
    latitude = _ask_float("Latitude")
    longitude = _ask_float("Longitude")
    coordinates = [latitude, longitude]

    meteo_mode = _ask_str(
        "Meteorological mode",
        allowed=["historical", "current", "forecast"],
    )

    meteo_date = None
    meteo_hour = None

    if meteo_mode in ["historical", "forecast"]:
        meteo_date = _ask_str("Date (YYYY-MM-DD)")
        meteo_hour = _ask_int("Hour (0-23)", min_value=0, max_value=23)

    use_mapbox = _ask_bool("Use Mapbox background?", default=False)

    # -------------------------------------------------
    # Build config
    # -------------------------------------------------
    config = SimulationConfig(
        max_x=max_x,
        max_y=max_y,
        grid_resolution=grid_resolution,
        evaluation_height_z=evaluation_height_z,
        emission_rate_Q=emission_rate_Q,
        emission_height_H=emission_height_H,
        wind_components=wind_components,
        dispersion_model=dispersion_model,
        stability_index=stability_index,
        colormap_index=colormap_index,
        radionuclide=radionuclide,
        population=population,
        exposure_pathway=exposure_pathway,
        absorption_type=absorption_type,
        age_group=age_group,
        amad=amad,
        ingestion_key=ingestion_key,
        gender_subject=gender_subject,
        breathing_mode=breathing_mode,
        particle_shape_factor=particle_shape_factor,
        particle_density_g_cm3=particle_density_g_cm3,
        coordinates=coordinates,
        meteo_mode=meteo_mode,
        meteo_date=meteo_date,
        meteo_hour=meteo_hour,
        exposure_time_h=exposure_time_h,
        use_regional_sf=use_regional_sf,
        use_mapbox=use_mapbox,
    )

    return config


def main():
    config = build_config_interactively()
    results = run_simulation(config)

    print("\nSimulation completed successfully.")
    print(f"Results saved in: {results.results_dir}")

    print("\nReturned Python API objects:")
    print(f"DataFrames: {list(results.dataframes.keys())}")
    print(f"Arrays: {list(results.arrays.keys())}")
    print(f"Figures: {list(results.figures.keys())}")
    print(f"Metadata: {list(results.metadata.keys())}")


if __name__ == "__main__":
    main()