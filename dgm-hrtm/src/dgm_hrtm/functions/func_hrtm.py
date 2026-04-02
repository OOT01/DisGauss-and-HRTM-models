import numpy as np

from dgm_hrtm.hrtm import lib_HRTM as hrtm


# =========================================================
# HRTM FUNCTIONS
# =========================================================
# These functions act as workflow helpers between the Gaussian
# dispersion model and the HRTM library.
#
# Version 2.2:
#   - full-map HRTM support
#   - all-activity HRTM support
#   - subject-aware HRTM inputs
#   - breathing-mode support
#   - aerosol metadata support
#   - optional regional sf_* modifiers
# =========================================================


def extract_map_maxima(C, X, Y, activity_results):
    """
    Extract spatial maxima for one HRTM activity result over the full map.
    """
    if C is None or X is None or Y is None:
        raise ValueError("C, X and Y must be provided.")

    if activity_results is None:
        raise ValueError("activity_results must be provided.")

    if C.shape != X.shape or C.shape != Y.shape:
        raise ValueError("C, X and Y must have the same shape.")

    if np.all(np.isnan(C)):
        raise ValueError("Concentration field contains only NaN values.")

    intake_map = activity_results["intake_map_bq"]
    deposited_maps = activity_results["deposited_activity_maps_bq"]

    idx_cmax = np.unravel_index(np.nanargmax(C), C.shape)
    idx_intake_max = np.unravel_index(np.nanargmax(intake_map), intake_map.shape)
    idx_resp_max = np.unravel_index(
        np.nanargmax(deposited_maps["respiratory_total"]),
        deposited_maps["respiratory_total"].shape,
    )

    summary = {
        "concentration_max": {
            "index": idx_cmax,
            "x_m": float(X[idx_cmax]),
            "y_m": float(Y[idx_cmax]),
            "value_bq_m3": float(C[idx_cmax]),
        },
        "intake_max": {
            "index": idx_intake_max,
            "x_m": float(X[idx_intake_max]),
            "y_m": float(Y[idx_intake_max]),
            "value_bq": float(intake_map[idx_intake_max]),
        },
        "respiratory_total_max": {
            "index": idx_resp_max,
            "x_m": float(X[idx_resp_max]),
            "y_m": float(Y[idx_resp_max]),
            "value_bq": float(deposited_maps["respiratory_total"][idx_resp_max]),
        },
        "region_maxima": {},
    }

    for region_name, region_map in deposited_maps.items():
        idx_region_max = np.unravel_index(np.nanargmax(region_map), region_map.shape)
        summary["region_maxima"][region_name] = {
            "index": idx_region_max,
            "x_m": float(X[idx_region_max]),
            "y_m": float(Y[idx_region_max]),
            "value_bq": float(region_map[idx_region_max]),
        }

    return summary


def run_hrtm_on_map(
    C,
    X,
    Y,
    amad="1um",
    exposure_time_h=1.0,
    breathing_rate_m3_h=None,
    age_group="adult",
    gender_subject="male",
    breathing_mode="nasal",
    particle_shape_factor=1.5,
    particle_density_g_cm3=3.0,
    activity_label=None,
    use_regional_sf=False,
):
    """
    Run HRTM over the full concentration map for one activity case.
    """
    if C is None or X is None or Y is None:
        raise ValueError("C, X and Y must be provided.")

    if C.shape != X.shape or C.shape != Y.shape:
        raise ValueError("C, X and Y must have the same shape.")

    hrtm_map_result = hrtm.run_hrtm_map(
        concentration_map_bq_m3=C,
        amad=amad,
        exposure_time_h=exposure_time_h,
        breathing_rate_m3_h=breathing_rate_m3_h,
        age_group=age_group,
        gender_subject=gender_subject,
        breathing_mode=breathing_mode,
        particle_shape_factor=particle_shape_factor,
        particle_density_g_cm3=particle_density_g_cm3,
        activity_label=activity_label,
        use_regional_sf=use_regional_sf,
    )

    maxima_summary = extract_map_maxima(C, X, Y, hrtm_map_result)

    return {
        "X_m": X,
        "Y_m": Y,
        **hrtm_map_result,
        "maxima_summary": maxima_summary,
    }


def run_hrtm_on_map_all_activities(
    C,
    X,
    Y,
    amad="1um",
    exposure_time_h=1.0,
    breathing_rates_m3_h=None,
    age_group="adult",
    gender_subject="male",
    breathing_mode="nasal",
    particle_shape_factor=1.5,
    particle_density_g_cm3=3.0,
    use_regional_sf=False,
):
    """
    Run HRTM over the full concentration map for all activity cases.
    """
    if C is None or X is None or Y is None:
        raise ValueError("C, X and Y must be provided.")

    if C.shape != X.shape or C.shape != Y.shape:
        raise ValueError("C, X and Y must have the same shape.")

    hrtm_all = hrtm.run_hrtm_map_all_activities(
        concentration_map_bq_m3=C,
        amad=amad,
        exposure_time_h=exposure_time_h,
        breathing_rates_m3_h=breathing_rates_m3_h,
        age_group=age_group,
        gender_subject=gender_subject,
        breathing_mode=breathing_mode,
        particle_shape_factor=particle_shape_factor,
        particle_density_g_cm3=particle_density_g_cm3,
        use_regional_sf=use_regional_sf,
    )

    results_with_maxima = {}
    for activity, activity_result in hrtm_all["results_by_activity"].items():
        maxima_summary = extract_map_maxima(C, X, Y, activity_result)

        results_with_maxima[activity] = {
            **activity_result,
            "maxima_summary": maxima_summary,
        }

    return {
        "mode": "map_all_activities",
        "X_m": X,
        "Y_m": Y,
        "amad": hrtm_all["amad"],
        "exposure_time_h": hrtm_all["exposure_time_h"],
        "age_group_input": hrtm_all.get("age_group_input"),
        "gender_subject_input": hrtm_all.get("gender_subject_input"),
        "breathing_mode": hrtm_all.get("breathing_mode"),
        "particle_shape_factor": hrtm_all.get("particle_shape_factor"),
        "particle_density_g_cm3": hrtm_all.get("particle_density_g_cm3"),
        "use_regional_sf": hrtm_all.get("use_regional_sf"),
        "activities": hrtm_all["activities"],
        "breathing_rates_m3_h": hrtm_all["breathing_rates_m3_h"],
        "results_by_activity": results_with_maxima,
    }


def save_hrtm_activity_plot(
    hrtm_results_all_activities,
    run_dir,
    filename="hrtm_activity_comparison.png",
    quantity_key="respiratory_total",
):
    """
    Save one combined comparison plot for all activity cases.
    """
    return hrtm.save_hrtm_activity_comparison_plot(
        hrtm_results_all_activities=hrtm_results_all_activities,
        run_dir=run_dir,
        filename=filename,
        quantity_key=quantity_key,
    )


def save_hrtm_regional_plot(
    hrtm_results_all_activities,
    run_dir,
    filename="hrtm_regional_activity_barplot.png",
):
    """
    Save the grouped regional comparison figure.
    """
    return hrtm.save_hrtm_regional_activity_barplot(
        hrtm_results_all_activities=hrtm_results_all_activities,
        run_dir=run_dir,
        filename=filename,
    )