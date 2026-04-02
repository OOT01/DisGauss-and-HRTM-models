import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from dgm_hrtm.functions import func_hrtm as fhrtm
from dgm_hrtm.functions.func_utils_map import add_mapbox_background


def convert_half_life_to_lambda(half_life, unit="years"):
    if unit == "years":
        return np.log(2) / (half_life * 365.25 * 24 * 3600)
    elif unit == "days":
        return np.log(2) / (half_life * 24 * 3600)
    elif unit == "hours":
        return np.log(2) / (half_life * 3600)
    elif unit == "minutes":
        return np.log(2) / (half_life * 60)
    elif unit == "seconds":
        return np.log(2) / half_life
    elif unit == "stable":
        return 0.0
    else:
        raise ValueError(f"Unknown unit: {unit}")


def concentracion_gaussiana(x, y, z, Q, u_total, H, sigma_y, sigma_z, lambda_):
    term1 = Q / (2 * np.pi * sigma_y * sigma_z * u_total)
    term2 = np.exp(-y**2 / (2 * sigma_y**2))
    term3 = np.exp(-(z - H)**2 / (2 * sigma_z**2)) + np.exp(
        -(z + H)**2 / (2 * sigma_z**2)
    )
    return term1 * term2 * term3 * np.exp(-lambda_ * np.sqrt(x**2 + y**2) / u_total)


def _cfg(config, key):
    """Read field from dataclass/object or dict."""
    if isinstance(config, dict):
        return config[key]
    return getattr(config, key)


def MododeloContaminacionGaussiana(
    config,
    df_rad,
    dcf_data,
    results_dir,
    meteo=None,
    map_cfg=None,
):
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # -------------------------------
    # Read configuration
    # -------------------------------
    max_x = _cfg(config, "max_x")
    max_y = _cfg(config, "max_y")
    grid_resolution = _cfg(config, "grid_resolution")
    z = _cfg(config, "evaluation_height_z")

    Q = _cfg(config, "emission_rate_Q")
    H = _cfg(config, "emission_height_H")

    dispersion_model = _cfg(config, "dispersion_model")
    stability_index = int(_cfg(config, "stability_index"))
    colormap_index = _cfg(config, "colormap_index")

    radionuclide = _cfg(config, "radionuclide")
    population = _cfg(config, "population")
    exposure_pathway = _cfg(config, "exposure_pathway")
    absorption_type = _cfg(config, "absorption_type")
    amad = _cfg(config, "amad")
    ingestion_key = _cfg(config, "ingestion_key")
    age_group = _cfg(config, "age_group")
    exposure_time_h = _cfg(config, "exposure_time_h")
    gender_subject = _cfg(config, "gender_subject")
    breathing_mode = _cfg(config, "breathing_mode")
    particle_shape_factor = _cfg(config, "particle_shape_factor")
    particle_density_g_cm3 = _cfg(config, "particle_density_g_cm3")
    use_regional_sf = _cfg(config, "use_regional_sf")

    coords = _cfg(config, "coordinates")
    mode = _cfg(config, "meteo_mode")
    date = _cfg(config, "meteo_date")
    hour = _cfg(config, "meteo_hour")

    wind_components = _cfg(config, "wind_components")
    u_x = wind_components[0]
    u_y = wind_components[1]

    # -------------------------------
    # Wind parameters
    # -------------------------------
    theta = np.arctan2(u_y, u_x)
    u_total = np.sqrt(u_x**2 + u_y**2)

    # -------------------------------
    # Original domain
    # -------------------------------
    x = np.linspace(-int(max_x), int(max_x), int(grid_resolution))
    y = np.linspace(-int(max_y), int(max_y), int(grid_resolution))
    X, Y = np.meshgrid(x, y)

    # -------------------------------
    # Coordinate rotation
    # -------------------------------
    X_prime = X * np.cos(theta) + Y * np.sin(theta)
    Y_prime = -X * np.sin(theta) + Y * np.cos(theta)

    # -------------------------------
    # Dispersion parameters Model of A. Pasquill-Gifford (P-G)
    # -------------------------------
    A_Y = [0.22, 0.16, 0.11, 0.08, 0.06, 0.04]
    A_Z = [0.20, 0.12, 0.08, 0.06, 0.03, 0.016]
    B_Z = [0.0, 0.0, 0.0002, 0.0015, 0.0003, 0.0003]
    C_Z = [0.0, 0.0, 0.5, 0.5, 1, 1]

    # Briggs parameters – Rural Region
    a_r = [0.22, 0.16, 0.11, 0.08, 0.06, 0.04]
    b_r = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    c_r = [0.20, 0.12, 0.08, 0.06, 0.03, 0.016]
    d_r = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    # Briggs parameters – Urban Region
    a_u = [0.32, 0.32, 0.22, 0.16, 0.11, 0.11]
    b_u = [0.78, 0.78, 0.78, 0.78, 0.78, 0.78]
    c_u = [0.24, 0.24, 0.20, 0.14, 0.08, 0.08]
    d_u = [0.78, 0.78, 0.78, 0.78, 0.78, 0.78]

    texto = [
        "very unstable",
        "unstable",
        "slightly unstable",
        "neutral",
        "slightly stable",
        "stable",
    ]

    X_pos = np.where(X_prime > 0, X_prime, np.nan)

    if dispersion_model == "Briggs_r":
        sigma_y = a_r[stability_index] * X_pos**b_r[stability_index]
        sigma_z = c_r[stability_index] * X_pos**d_r[stability_index]
    elif dispersion_model == "Briggs_u":
        sigma_y = a_u[stability_index] * X_pos**b_u[stability_index]
        sigma_z = c_u[stability_index] * X_pos**d_u[stability_index]
    else:
        sigma_y = A_Y[stability_index] * X_pos * (1 + 0.0001 * X_pos) ** (-0.5)
        sigma_z = A_Z[stability_index] * X_pos * (
            1 + B_Z[stability_index] * X_pos
        ) ** (-C_Z[stability_index])

    sigma_y[X_prime <= 0] = np.nan
    sigma_z[X_prime <= 0] = np.nan

    # -------------------------------
    # Lambda from CSV
    # -------------------------------
    fila = df_rad.loc[df_rad["Radionuclido"] == radionuclide]
    if fila.empty:
        raise ValueError(f"Radionuclide '{radionuclide}' not found in radionuclidos.csv")

    half_life = float(fila.iloc[0]["Half_life_value"])
    unit = str(fila.iloc[0]["Unit"]).strip().lower()
    lambda_ = convert_half_life_to_lambda(half_life, unit)

    # -------------------------------
    # Concentration calculation
    # -------------------------------
    mask_downwind = X_prime > 0
    sigma_y[~mask_downwind] = 1.0
    sigma_z[~mask_downwind] = 1.0

    C = concentracion_gaussiana(
        X_prime,
        Y_prime,
        z,
        Q,
        u_total,
        H,
        sigma_y,
        sigma_z,
        lambda_,
    )

    # Set upwind region to zero concentration
    C[~mask_downwind] = 0.0

    # -------------------------------
    # Graphic Model Selection
    # -------------------------------
    cmap_options = [
        "viridis",
        "plasma",
        "inferno",
        "magma",
        "cividis",
        "Greys",
        "Blues",
        "Oranges",
        "Purples",
        "YlGnBu",
        "BuGn",
        "coolwarm",
        "seismic",
        "bwr",
        "PiYG",
        "hsv",
        "twilight",
        "tab10",
        "tab20",
        "Set1",
        "Set2",
    ]

    # -------------------------------
    # DCF selection
    # -------------------------------
    nucl = radionuclide
    ruta_dcf = exposure_pathway
    tipo = absorption_type

    if nucl not in dcf_data:
        raise ValueError(f"No DCF entry for '{nucl}'")

    if population == "worker":
        if ruta_dcf == "inhalation":
            key = f"e_{amad}"
            DCF = float(dcf_data[nucl]["inhalation"][tipo][key])
        elif ruta_dcf == "ingestion":
            DCF = float(dcf_data[nucl]["ingestion"][ingestion_key]["e"])
        else:
            raise ValueError("Invalid exposure pathway")

    elif population == "public":
        if ruta_dcf == "inhalation":
            DCF = float(dcf_data[nucl]["inhalation"][tipo][age_group]["e"])
        elif ruta_dcf == "ingestion":
            DCF = float(dcf_data[nucl]["ingestion"][ingestion_key][age_group]["e"])
        else:
            raise ValueError("Invalid exposure pathway")

    else:
        raise ValueError("Invalid population")

    # -------------------------------
    # Create results directory for this run
    # -------------------------------
    base_run_name = f"{nucl}_{dispersion_model}_{population}_{ruta_dcf}_{tipo}_{amad}"
    run_name = base_run_name
    run_dir = os.path.join(results_dir, run_name)

    run_counter = 1
    while os.path.exists(run_dir):
        run_name = f"{base_run_name}_run_{run_counter:02d}"
        run_dir = os.path.join(results_dir, run_name)
        run_counter += 1

    os.makedirs(run_dir, exist_ok=True)

    # -------------------------------
    # Respiratory rate m3/h
    # -------------------------------
    # Legacy breathing rates kept only for the plume-dose plots below.
    BR = [0.6, 1.2, 1.8, 3.0]  # Rest, light activity, moderate activity, heavy activity

    # -------------------------------
    # HRTM (full map, all activity cases)
    # -------------------------------
    hrtm_results = None
    figure_paths = {}

    if ruta_dcf == "inhalation" and population == "public":
        # Breathing rates for HRTM are resolved inside lib_HRTM from
        # hrtm_subjects_breathing.json according to Table 15.
        hrtm_results = fhrtm.run_hrtm_on_map_all_activities(
            C=C,
            X=X,
            Y=Y,
            amad=amad,
            exposure_time_h=exposure_time_h,
            breathing_rates_m3_h=None,
            age_group=age_group,
            gender_subject=gender_subject,
            breathing_mode=breathing_mode,
            particle_shape_factor=particle_shape_factor,
            particle_density_g_cm3=particle_density_g_cm3,
            use_regional_sf=use_regional_sf,
        )

        # -------------------------------
        # Save HRTM result to TXT file
        # -------------------------------
        hrtm_file = os.path.join(run_dir, "hrtm_results_all_activities.txt")

        with open(hrtm_file, "w", encoding="utf-8") as f:
            f.write("============================================================\n")
            f.write("     HRTM RESULTS (FULL MAP - ALL ACTIVITY CASES)\n")
            f.write("============================================================\n\n")

            f.write(f"Run timestamp                      : {run_timestamp}\n")
            f.write(f"AMAD                               : {hrtm_results['amad']}\n")
            f.write(f"Exposure time (h)                  : {hrtm_results['exposure_time_h']}\n")
            f.write(f"Age group                          : {hrtm_results.get('age_group_input')}\n")
            f.write(f"Gender subject                     : {hrtm_results.get('gender_subject_input')}\n")
            f.write(f"Breathing mode                     : {hrtm_results.get('breathing_mode')}\n")
            f.write(f"Particle shape factor              : {hrtm_results.get('particle_shape_factor')}\n")
            f.write(f"Particle density (g/cm^3)          : {hrtm_results.get('particle_density_g_cm3')}\n")
            f.write(f"Use regional sf modifiers          : {hrtm_results.get('use_regional_sf')}\n")
            f.write(f"Activities                         : {', '.join(hrtm_results['activities'])}\n\n")

            for activity in hrtm_results["activities"]:
                result = hrtm_results["results_by_activity"][activity]
                maxima = result["maxima_summary"]
                deposited_maps = result["deposited_activity_maps_bq"]

                f.write("------------------------------------------------------------\n")
                f.write(f"ACTIVITY CASE                      : {activity.upper()}\n")
                f.write("------------------------------------------------------------\n")
                f.write(f"Breathing rate (m³/h)              : {result['breathing_rate_m3_h']}\n")
                f.write(f"Resolved age group                 : {result.get('age_group_resolved')}\n")
                f.write(f"Resolved gender                    : {result.get('gender_resolved')}\n")
                f.write(f"Breathing mode                     : {result.get('breathing_mode')}\n")
                f.write(f"Nasal fraction Fn                  : {result.get('fn_nasal_fraction')}\n")
                f.write(f"Particle shape factor              : {result.get('particle_shape_factor')}\n")
                f.write(f"Particle density (g/cm^3)          : {result.get('particle_density_g_cm3')}\n")
                f.write(f"Use regional sf modifiers          : {result.get('use_regional_sf')}\n")
                f.write(f"Maximum air concentration (Bq/m³)  : {maxima['concentration_max']['value_bq_m3']}\n")
                f.write(f"At X (m)                           : {maxima['concentration_max']['x_m']}\n")
                f.write(f"At Y (m)                           : {maxima['concentration_max']['y_m']}\n\n")

                f.write(f"Maximum inhaled activity (Bq)      : {maxima['intake_max']['value_bq']}\n")
                f.write(f"At X (m)                           : {maxima['intake_max']['x_m']}\n")
                f.write(f"At Y (m)                           : {maxima['intake_max']['y_m']}\n\n")

                f.write(f"Maximum respiratory deposition (Bq): {maxima['respiratory_total_max']['value_bq']}\n")
                f.write(f"At X (m)                           : {maxima['respiratory_total_max']['x_m']}\n")
                f.write(f"At Y (m)                           : {maxima['respiratory_total_max']['y_m']}\n\n")

                f.write("Deposition fractions\n")
                f.write("............................................................\n")
                for k, v in result["deposition_fractions"].items():
                    f.write(f"{k:<35} : {v}\n")

                f.write(f"\nFraction deposited in respiratory tract : {result['respiratory_fraction']}\n\n")

                f.write("Maximum deposited activity by region (Bq)\n")
                f.write("............................................................\n")
                f.write(f"ET1                              : {np.nanmax(deposited_maps['ET1'])}\n")
                f.write(f"ET2                              : {np.nanmax(deposited_maps['ET2'])}\n")
                f.write(f"BB                               : {np.nanmax(deposited_maps['BB'])}\n")
                f.write(f"bb                               : {np.nanmax(deposited_maps['bb'])}\n")
                f.write(f"AI                               : {np.nanmax(deposited_maps['AI'])}\n\n")
                f.write(f"Extrathoracic total              : {np.nanmax(deposited_maps['extrathoracic_total'])}\n")
                f.write(f"Thoracic total                   : {np.nanmax(deposited_maps['thoracic_total'])}\n")
                f.write(f"Total deposited (respiratory)    : {np.nanmax(deposited_maps['respiratory_total'])}\n\n")

        # -------------------------------
        # Save HRTM summary plots
        # -------------------------------
        hrtm_activity_plot_path = os.path.join(run_dir, "hrtm_activity_comparison.png")
        hrtm_regional_plot_path = os.path.join(run_dir, "hrtm_regional_activity_barplot.png")

        fhrtm.save_hrtm_activity_plot(
            hrtm_results_all_activities=hrtm_results,
            run_dir=run_dir,
            filename="hrtm_activity_comparison.png",
            quantity_key="respiratory_total",
        )

        fhrtm.save_hrtm_regional_plot(
            hrtm_results_all_activities=hrtm_results,
            run_dir=run_dir,
            filename="hrtm_regional_activity_barplot.png",
        )

        figure_paths["hrtm_activity_comparison"] = hrtm_activity_plot_path
        figure_paths["hrtm_regional_activity_barplot"] = hrtm_regional_plot_path

    elif ruta_dcf == "inhalation" and population == "worker":
        print("Note: HRTM skipped for worker mode. Gaussian dispersion results only.")

    # -------------------------------
    # Save parameters to TXT file
    # -------------------------------
    param_file = os.path.join(run_dir, "parameters.txt")

    def write_param(f, name, value, unit_text=""):
        if unit_text:
            f.write(f"{name:<35} : {value} {unit_text}\n")
        else:
            f.write(f"{name:<35} : {value}\n")

    with open(param_file, "w", encoding="utf-8") as f:
        f.write("\n")
        f.write("=============================================\n")
        f.write("      GAUSSIAN ATMOSPHERIC DISPERSION RUN\n")
        f.write("=============================================\n\n")

        write_param(f, "Run timestamp", run_timestamp)
        f.write("\n")

        f.write("RADIONUCLIDE CONFIGURATION\n")
        f.write("---------------------------------------------\n")
        write_param(f, "Nuclide", nucl)
        write_param(f, "Population", population)
        write_param(f, "Exposure pathway", ruta_dcf)
        write_param(f, "Absorption type", tipo)
        write_param(f, "AMAD", amad)
        write_param(f, "Age group", age_group)
        write_param(f, "Gender subject", gender_subject)
        write_param(f, "Breathing mode", breathing_mode)
        write_param(f, "Particle shape factor", particle_shape_factor)
        write_param(f, "Particle density", particle_density_g_cm3, "g/cm^3")
        write_param(f, "Use regional sf modifiers", use_regional_sf)
        write_param(f, "Ingestion key", ingestion_key)
        f.write("\n")

        f.write("DOSE COEFFICIENT\n")
        f.write("---------------------------------------------\n")
        write_param(f, "DCF", f"{DCF:.3e}", "Sv/Bq")
        f.write("\n")

        f.write("GAUSSIAN PLUME MODEL\n")
        f.write("---------------------------------------------\n")
        write_param(f, "Dispersion model", dispersion_model)
        write_param(f, "Maximum X", max_x, "m")
        write_param(f, "Maximum Y", max_y, "m")
        write_param(f, "Grid resolution", grid_resolution, "points")
        write_param(f, "Evaluation height Z", z, "m")
        write_param(f, "Emission rate Q", Q, "Bq/s")
        write_param(f, "Wind components [u_x, u_y]", wind_components, "m/s")
        write_param(f, "Wind magnitude u_total", f"{u_total:.6f}", "m/s")
        write_param(f, "Effective emission height H", H, "m")
        write_param(f, "Atmospheric stability index", stability_index)
        write_param(f, "Colormap index", colormap_index)
        write_param(f, "Exposure time for HRTM", exposure_time_h, "h")
        f.write("\n")

        f.write("RADIOACTIVE DECAY\n")
        f.write("---------------------------------------------\n")
        write_param(f, "Half-life value", half_life)
        write_param(f, "Half-life unit", unit)
        write_param(f, "Decay constant lambda", f"{lambda_:.6e}", "1/s")
        f.write("\n")

        if coords is not None or mode is not None or date is not None or hour is not None:
            f.write("METEOROLOGICAL CONFIGURATION\n")
            f.write("---------------------------------------------\n")
            write_param(f, "Coordinates", coords)
            write_param(f, "Mode", mode)
            if mode == "current":
                write_param(f, "Date", "-")
                write_param(f, "Hour", "-")
            else:
                write_param(f, "Date", date)
                write_param(f, "Hour", hour)
            f.write("\n")

        if meteo is not None:
            f.write("METEOROLOGICAL DATA (Open-Meteo ECMWF-IFS)\n")
            f.write("---------------------------------------------\n")

            if "time" in meteo:
                write_param(f, "Timestamp", meteo["time"])

            if "temperature_2m" in meteo:
                write_param(f, "Temperature 2m", meteo["temperature_2m"], "°C")

            if "relative_humidity_2m" in meteo:
                write_param(f, "Relative humidity 2m", meteo["relative_humidity_2m"], "%")

            if "surface_pressure" in meteo:
                write_param(f, "Surface pressure", meteo["surface_pressure"], "hPa")

            if "precipitation" in meteo:
                write_param(f, "Precipitation", meteo["precipitation"], "mm")

            if "cloud_cover" in meteo:
                write_param(f, "Cloud cover", meteo["cloud_cover"], "%")

            if "precipitation_probability" in meteo:
                write_param(f, "Precipitation probability", meteo["precipitation_probability"], "%")

            if "wind_speed_10m" in meteo:
                write_param(f, "Wind speed 10m", meteo["wind_speed_10m"], "m/s")

            if "wind_direction_10m" in meteo:
                write_param(f, "Wind direction 10m", meteo["wind_direction_10m"], "deg")

            if "u_x" in meteo and "u_y" in meteo:
                write_param(f, "Wind component u_x", meteo["u_x"], "m/s")
                write_param(f, "Wind component u_y", meteo["u_y"], "m/s")

    # Label for the exposed person/group
    if population == "worker":
        if age_group == "adult" and gender_subject in ["male", "female"]:
            subject_label = f"Worker ({gender_subject.capitalize()})"
        else:
            subject_label = "Worker"

    elif population == "public":
        if age_group == "adult":
            if gender_subject == "male":
                subject_label = "Adult Male"
            elif gender_subject == "female":
                subject_label = "Adult Female"
            else:
                subject_label = "Adult"

        elif age_group == "age_15y":
            if gender_subject == "male":
                subject_label = "15 y Male"
            elif gender_subject == "female":
                subject_label = "15 y Female"
            else:
                subject_label = "Child (15 y)"

        else:
            age_labels = {
                "infant": "Infant",
                "age_1y": "Child (1 y)",
                "age_5y": "Child (5 y)",
                "age_10y": "Child (10 y)",
                "age_15y": "Child (15 y)",
                "adult": "Adult",
            }
            subject_label = age_labels.get(age_group, "Public")
    else:
        subject_label = "Subject"

    # -------------------------------
    # Select plots according to population / age
    # -------------------------------
    if population == "public" and age_group in ["infant", "age_1y"]:
        field_list = [C, C * DCF * BR[0] * 24 * 1e6]

        titles = [
            "Gaussian Dispersion Contamination",
            f"Average {subject_label} at Rest",
        ]

        colorbar_labels = [
            "Activity Concentration (Bq/m³)",
            "Inhalation Dose (µSv)",
        ]

        field_names = [
            "concentration",
            "dose_rest",
        ]

    elif population == "public" and age_group == "age_5y":
        field_list = [
            C,
            C * DCF * BR[0] * 24 * 1e6,
            C * DCF * BR[1] * 24 * 1e6,
        ]

        titles = [
            "Gaussian Dispersion Contamination",
            f"Average {subject_label} at Rest",
            f"Average {subject_label} Light Activity",
        ]

        colorbar_labels = [
            "Activity Concentration (Bq/m³)",
            "Inhalation Dose (µSv)",
            "Inhalation Dose (µSv)",
        ]

        field_names = [
            "concentration",
            "dose_rest",
            "dose_light_activity",
        ]

    else:
        field_list = [
            C,
            C * DCF * BR[0] * 24 * 1e6,
            C * DCF * BR[1] * 24 * 1e6,
            C * DCF * BR[2] * 24 * 1e6,
        ]

        titles = [
            "Gaussian Dispersion Contamination",
            f"Average {subject_label} at Rest",
            f"Average {subject_label} Light Activity",
            f"Average {subject_label} Moderate Activity",
        ]

        colorbar_labels = [
            "Activity Concentration (Bq/m³)",
            "Inhalation Dose (µSv)",
            "Inhalation Dose (µSv)",
            "Inhalation Dose (µSv)",
        ]

        field_names = [
            "concentration",
            "dose_rest",
            "dose_light_activity",
            "dose_moderate_activity",
        ]

    # Colormaps only for the number of plots actually used
    cmaps = [
        cmap_options[int(colormap_index) + i] for i in range(len(field_list))
    ]

    # Figure size adapted to the number of rows
    n_rows = len(field_list)
    fig = plt.figure(figsize=(16, 5.5 * n_rows))

    for i, (field, title, cbar_label, cmap_sel) in enumerate(
        zip(field_list, titles, colorbar_labels, cmaps)
    ):
        # -------------------------
        # LEFT: 2D contour plot
        # -------------------------
        ax2d = fig.add_subplot(n_rows, 2, 2 * i + 1)

        # =========================================================
        # MAPBOX BACKGROUND
        # =========================================================
        if map_cfg is not None and map_cfg.enabled:
            try:
                lat0 = coords[0]
                lon0 = coords[1]

                x_min = float(np.min(X))
                x_max = float(np.max(X))
                y_min = float(np.min(Y))
                y_max = float(np.max(Y))

                add_mapbox_background(
                    ax=ax2d,
                    lat0_deg=lat0,
                    lon0_deg=lon0,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    map_cfg=map_cfg,
                    results_dir=run_dir,
                )

            except Exception as e:
                print(f"⚠️ Mapbox failed: {e}")

        # =========================================================
        # GAUSSIAN FIELD
        # =========================================================
        cp = ax2d.contourf(
            X,
            Y,
            field,
            levels=50,
            cmap=cmap_sel,
            alpha=0.65 if (map_cfg is not None and map_cfg.enabled) else 1.0,
            zorder=2,
        )

        fig.colorbar(cp, ax=ax2d, label=cbar_label, format="%.2e")

        ax2d.set_title(title, fontsize=14)
        ax2d.set_xlabel("x (m)")
        ax2d.set_ylabel("y (m)")
        ax2d.set_xlim(-max_x, max_x)
        ax2d.set_ylim(-max_y, max_y)
        ax2d.set_aspect("equal", adjustable="box")
        ax2d.grid(True)
        ax2d.plot(0, 0, "ro", markersize=5)

        # -------------------------
        # RIGHT: 3D surface plot
        # -------------------------
        ax3d = fig.add_subplot(n_rows, 2, 2 * i + 2, projection="3d")

        field_smooth = gaussian_filter(field, sigma=6)

        max_field = np.nanmax(field_smooth)

        if np.isfinite(max_field) and max_field > 0:
            threshold = 0.001 * max_field
            mask_plot = field_smooth > threshold
        else:
            mask_plot = np.zeros_like(field_smooth, dtype=bool)

        if np.any(mask_plot):
            iy, ix = np.where(mask_plot)

            pad = 15
            i_min = max(0, iy.min() - pad)
            i_max = min(field.shape[0], iy.max() + pad + 1)
            j_min = max(0, ix.min() - pad)
            j_max = min(field.shape[1], ix.max() + pad + 1)

            X_plot = X[i_min:i_max, j_min:j_max]
            Y_plot = Y[i_min:i_max, j_min:j_max]
            Z_plot = field_smooth[i_min:i_max, j_min:j_max]
        else:
            X_plot = X
            Y_plot = Y
            Z_plot = field_smooth

        z_top = 1.05 * np.nanmax(Z_plot) if np.nanmax(Z_plot) > 0 else 1.0

        surf = ax3d.plot_surface(
            X_plot,
            Y_plot,
            Z_plot,
            cmap=cmap_sel,
            edgecolor="none",
            linewidth=0,
            antialiased=True,
            shade=True,
            rcount=200,
            ccount=200,
        )

        # Upper projection
        ax3d.contourf(
            X_plot,
            Y_plot,
            Z_plot,
            zdir="z",
            offset=z_top,
            levels=30,
            cmap=cmap_sel,
            alpha=0.35,
        )

        fig.colorbar(
            surf,
            ax=ax3d,
            shrink=0.72,
            pad=0.08,
            aspect=18,
            label=cbar_label,
        )

        ax3d.set_title(title + " - 3D", fontsize=14, pad=14)
        ax3d.set_xlabel("X (m)", labelpad=8)
        ax3d.set_ylabel("Y (m)", labelpad=8)
        ax3d.set_zlabel(cbar_label, labelpad=8)

        ax3d.view_init(elev=28, azim=-58)

        try:
            ax3d.set_proj_type("ortho")
        except AttributeError:
            pass

        ax3d.set_zlim(0, z_top)

        ax3d.xaxis.pane.set_alpha(0.08)
        ax3d.yaxis.pane.set_alpha(0.08)
        ax3d.zaxis.pane.set_alpha(0.00)
        ax3d.grid(True, alpha=0.20)

    fig.suptitle(
        "Gaussian Dispersion with Oblique Wind\nAtmospheric State: "
        + texto[stability_index],
        fontsize=18,
        y=0.995,
    )

    combined_plot_path = os.path.join(run_dir, "plume_2D_3D_combined.png")

    plt.tight_layout(rect=[0, 0.01, 1, 0.985])
    fig.savefig(
        combined_plot_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    figure_paths["plume_2D_3D_combined"] = combined_plot_path
    figure_paths["parameters_txt"] = param_file

    # -------------------------------
    # Arrays for Python API
    # -------------------------------
    arrays = {
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
        "X_prime": X_prime,
        "Y_prime": Y_prime,
        "sigma_y": sigma_y,
        "sigma_z": sigma_z,
        "concentration": C,
    }

    for field_name, field in zip(field_names[1:], field_list[1:]):
        arrays[field_name] = field

    if hrtm_results is not None:
        arrays["hrtm_results"] = hrtm_results

    # -------------------------------
    # DataFrames for Python API
    # -------------------------------
    dataframes = {}

    concentration_df = pd.DataFrame(
        {
            "x_m": X.ravel(),
            "y_m": Y.ravel(),
            "concentration_bq_m3": C.ravel(),
        }
    )
    dataframes["concentration_map"] = concentration_df

    for field_name, field in zip(field_names[1:], field_list[1:]):
        df_name = f"{field_name}_map"
        value_column = f"{field_name}_value"
        dataframes[df_name] = pd.DataFrame(
            {
                "x_m": X.ravel(),
                "y_m": Y.ravel(),
                value_column: field.ravel(),
            }
        )

    # -------------------------------
    # Return structured results
    # -------------------------------
    return {
        "dataframes": dataframes,
        "arrays": arrays,
        "figures": figure_paths,
        "metadata": {
            "run_timestamp": run_timestamp,
            "run_dir": run_dir,
            "subject_label": subject_label,
            "stability_label": texto[stability_index],
            "dispersion_model": dispersion_model,
            "radionuclide": radionuclide,
            "population": population,
            "exposure_pathway": exposure_pathway,
            "absorption_type": absorption_type,
            "amad": amad,
            "age_group": age_group,
            "gender_subject": gender_subject,
            "breathing_mode": breathing_mode,
            "particle_shape_factor": particle_shape_factor,
            "particle_density_g_cm3": particle_density_g_cm3,
            "use_regional_sf": use_regional_sf,
            "dose_coefficient_sv_bq": DCF,
            "half_life_value": half_life,
            "half_life_unit": unit,
            "decay_constant_lambda": lambda_,
            "wind_components": wind_components,
            "u_total": u_total,
            "coordinates": coords,
            "meteo_mode": mode,
            "meteo_date": date,
            "meteo_hour": hour,
            "meteo": meteo,
        },
    }