import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import pandas as pd

from dgm_hrtm.configs.simulation_config import SimulationConfig
from dgm_hrtm.functions.func_main import main_experiment


@dataclass
class SimulationResults:
    """
    Structured return object for the Python API.
    """
    results_dir: Path
    dataframes: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)
    figures: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def get_results_dir(base_dir=None) -> Path:
    """
    Return the RESULTS directory.

    If base_dir is None, RESULTS is created in the current working directory.
    """
    if base_dir is None:
        results_dir = Path.cwd() / "RESULTS"
    else:
        results_dir = Path(base_dir) / "RESULTS"

    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def load_radionuclide_data() -> pd.DataFrame:
    """
    Load radionuclide data from the packaged CSV file.
    """
    rad_path = files("dgm_hrtm.data").joinpath("radionuclidos.csv")
    return pd.read_csv(rad_path, comment="#")


def load_dcf_data(population: str) -> dict:
    """
    Load the appropriate DCF database according to the selected population.
    """
    population = str(population).strip().lower()

    if population == "worker":
        dcf_path = files("dgm_hrtm.data").joinpath("dcf_workers.json")
    elif population == "public":
        dcf_path = files("dgm_hrtm.data").joinpath("dcf_public.json")
    else:
        raise ValueError(f"Unknown population: {population}")

    with open(dcf_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_config(config, dcf_data):
    """
    Validate the logical consistency of the selected configuration against
    the loaded DCF database before running the simulation.
    """
    population = str(config.population).strip().lower()
    radionuclide = config.radionuclide
    pathway = config.exposure_pathway

    if radionuclide not in dcf_data:
        raise ValueError(
            f"Radionuclide '{radionuclide}' was not found in the {population} DCF database."
        )

    nucl_data = dcf_data[radionuclide]

    if pathway not in ["inhalation", "ingestion"]:
        raise ValueError(f"Unknown exposure pathway: {pathway}")

    if pathway == "inhalation":
        absorption_type = config.absorption_type
        if absorption_type is None:
            raise ValueError(
                "Absorption type is required for inhalation exposure."
            )

        inhalation_data = nucl_data.get("inhalation", {})
        if not inhalation_data:
            raise ValueError(
                f"Radionuclide '{radionuclide}' does not contain inhalation data."
            )

        if absorption_type not in inhalation_data:
            available = list(inhalation_data.keys())
            raise ValueError(
                f"Absorption type '{absorption_type}' is not available for radionuclide "
                f"'{radionuclide}'. Available types: {', '.join(available)}"
            )

        selected_inhalation_data = inhalation_data[absorption_type]

        if population == "worker":
            if config.amad is None:
                raise ValueError(
                    "AMAD is required for worker inhalation exposure."
                )

            amad_key = f"e_{config.amad}"
            available = list(selected_inhalation_data.keys())

            if amad_key not in selected_inhalation_data:
                raise ValueError(
                    f"AMAD '{config.amad}' is not available for radionuclide "
                    f"'{radionuclide}' with absorption type '{absorption_type}'. "
                    f"Available keys: {', '.join(available)}"
                )

        elif population == "public":
            if config.age_group is None:
                raise ValueError(
                    "Age group is required for public inhalation exposure."
                )

            available = list(selected_inhalation_data.keys())

            if config.age_group not in selected_inhalation_data:
                raise ValueError(
                    f"Age group '{config.age_group}' is not available for radionuclide "
                    f"'{radionuclide}' with absorption type '{absorption_type}'. "
                    f"Available groups: {', '.join(available)}"
                )

    elif pathway == "ingestion":
        ingestion_key = config.ingestion_key
        if ingestion_key is None:
            raise ValueError(
                "Ingestion key is required for ingestion exposure."
            )

        ingestion_data = nucl_data.get("ingestion", {})
        if not ingestion_data:
            raise ValueError(
                f"Radionuclide '{radionuclide}' does not contain ingestion data."
            )

        if ingestion_key not in ingestion_data:
            available = list(ingestion_data.keys())
            raise ValueError(
                f"Ingestion key '{ingestion_key}' is not available for radionuclide "
                f"'{radionuclide}'. Available keys: {', '.join(available)}"
            )

        selected_ingestion_data = ingestion_data[ingestion_key]

        if population == "worker":
            if "e" not in selected_ingestion_data:
                available = list(selected_ingestion_data.keys())
                raise ValueError(
                    f"Ingestion entry '{ingestion_key}' for radionuclide '{radionuclide}' "
                    f"does not contain worker dose coefficient key 'e'. "
                    f"Available keys: {', '.join(available)}"
                )

        elif population == "public":
            if config.age_group is None:
                raise ValueError(
                    "Age group is required for public ingestion exposure."
                )

            available = list(selected_ingestion_data.keys())

            if config.age_group not in selected_ingestion_data:
                raise ValueError(
                    f"Age group '{config.age_group}' is not available for radionuclide "
                    f"'{radionuclide}' with ingestion key '{ingestion_key}'. "
                    f"Available groups: {', '.join(available)}"
                )

            public_age_data = selected_ingestion_data[config.age_group]
            if "e" not in public_age_data:
                available = list(public_age_data.keys())
                raise ValueError(
                    f"Ingestion entry '{ingestion_key}' for radionuclide '{radionuclide}' "
                    f"and age group '{config.age_group}' does not contain key 'e'. "
                    f"Available keys: {', '.join(available)}"
                )


def _build_config_from_record(record):
    """
    Build a SimulationConfig from a dict-like record.
    """
    return SimulationConfig(
        max_x=record["max_x"],
        max_y=record["max_y"],
        grid_resolution=record["grid_resolution"],
        evaluation_height_z=record["evaluation_height_z"],
        emission_rate_Q=record["emission_rate_Q"],
        emission_height_H=record["emission_height_H"],
        wind_components=record.get("wind_components", [0.0, 0.0]),
        dispersion_model=record["dispersion_model"],
        stability_index=record["stability_index"],
        colormap_index=record["colormap_index"],
        radionuclide=record["radionuclide"],
        population=record["population"],
        exposure_pathway=record["exposure_pathway"],
        absorption_type=record.get("absorption_type"),
        age_group=record.get("age_group"),
        amad=record.get("amad"),
        ingestion_key=record.get("ingestion_key"),
        gender_subject=record.get("gender_subject"),
        breathing_mode=record["breathing_mode"],
        particle_shape_factor=record["particle_shape_factor"],
        particle_density_g_cm3=record["particle_density_g_cm3"],
        coordinates=record["coordinates"],
        meteo_mode=record["meteo_mode"],
        meteo_date=record.get("meteo_date"),
        meteo_hour=record.get("meteo_hour"),
        exposure_time_h=record["exposure_time_h"],
        use_regional_sf=record["use_regional_sf"],
        use_mapbox=record["use_mapbox"],
    )


def run_simulation(config, base_dir=None):
    """
    Run one full Gaussian dispersion + HRTM simulation.

    Parameters
    ----------
    config
        SimulationConfig instance.
    base_dir : str | Path | None
        Base directory where RESULTS will be created.
        If None, RESULTS is created in the current working directory.

    Returns
    -------
    SimulationResults
        Structured simulation outputs for the Python API.
    """
    results_dir = get_results_dir(base_dir=base_dir)
    df_rad = load_radionuclide_data()
    dcf_data = load_dcf_data(config.population)

    validate_config(config, dcf_data)

    experiment_results = main_experiment(
        config=config,
        df_rad=df_rad,
        dcf_data=dcf_data,
        results_dir=str(results_dir),
    )

    if isinstance(experiment_results, dict):
        return SimulationResults(
            results_dir=results_dir,
            dataframes=experiment_results.get("dataframes", {}),
            arrays=experiment_results.get("arrays", {}),
            figures=experiment_results.get("figures", {}),
            metadata=experiment_results.get("metadata", {}),
        )

    return SimulationResults(results_dir=results_dir)


def run_simulations_from_records(records, base_dir=None):
    """
    Run multiple simulations from a list of dict-like records.

    Parameters
    ----------
    records : list[dict]
        Each record contains the fields required to build a SimulationConfig.
    base_dir : str | Path | None
        Base directory where RESULTS will be created.

    Returns
    -------
    list[SimulationResults]
        One SimulationResults object per input record.
    """
    results = []

    for i, record in enumerate(records):
        try:
            config = _build_config_from_record(record)
            sim_results = run_simulation(config, base_dir=base_dir)
            results.append(sim_results)
        except Exception as e:
            raise ValueError(f"Error in record index {i}: {e}") from e

    return results


def run_simulations_from_dataframe(df_inputs, base_dir=None):
    """
    Run multiple simulations from a pandas DataFrame.

    Parameters
    ----------
    df_inputs : pandas.DataFrame
        Each row contains the fields required to build a SimulationConfig.
    base_dir : str | Path | None
        Base directory where RESULTS will be created.

    Returns
    -------
    list[SimulationResults]
        One SimulationResults object per DataFrame row.
    """
    if not isinstance(df_inputs, pd.DataFrame):
        raise TypeError("df_inputs must be a pandas DataFrame.")

    records = df_inputs.to_dict(orient="records")
    return run_simulations_from_records(records, base_dir=base_dir)