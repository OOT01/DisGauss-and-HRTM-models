import os
from importlib.resources import files

import matplotlib.pyplot as plt
import numpy as np

from dgm_hrtm.functions.func_hrtm_deposition import (
    get_breathing_params,
    get_regional_deposition_fractions,
)


# -----------------------------------------------------------------
# HRTM
# -----------------------------------------------------------------
# Main upgrade:
#   - regional deposition is handled through:
#       dgm_hrtm.functions.func_hrtm_deposition
#   - subject anatomy, breathing parameters and Fn are loaded from:
#       dgm_hrtm.data/hrtm_subjects_breathing.json
#
# This library acts as:
#   - map-level intake/deposition driver
#   - plotting/output utility
#   - interface layer between plume concentration maps and the
#     subject-aware HRTM deposition model
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# PACKAGE DATA PATH
# -----------------------------------------------------------------
HRTM_SUBJECTS_JSON_PATH = str(
    files("dgm_hrtm.data").joinpath("hrtm_subjects_breathing.json")
)


# -----------------------------------------------------------------
# INPUT NORMALISATION
# -----------------------------------------------------------------
def _normalise_amad(amad):
    """Return standard AMAD label used internally."""
    if amad is None:
        raise ValueError("AMAD cannot be None.")

    amad_str = str(amad).strip().lower().replace(" ", "")

    if amad_str in {"1um", "1", "1.0", "1µm", "1μm"}:
        return "1um"
    if amad_str in {"5um", "5", "5.0", "5µm", "5μm"}:
        return "5um"

    raise ValueError(f"Unsupported AMAD value: {amad}")


def _normalise_age_group(age_group):
    """Normalise age-group labels to internal JSON keys."""
    if age_group is None:
        raise ValueError("age_group cannot be None for HRTM subject resolution.")

    age_str = str(age_group).strip().lower()

    mapping = {
        "adult": "adult",
        "age_15y": "age_15y",
        "15y": "age_15y",
        "age15y": "age_15y",
        "age_10y": "age_10y",
        "10y": "age_10y",
        "age10y": "age_10y",
        "age_5y": "age_5y",
        "5y": "age_5y",
        "age5y": "age_5y",
        "age_1y": "age_1y",
        "1y": "age_1y",
        "age1y": "age_1y",
        "infant": "infant",
        "3mo": "infant",
        "age_3mo": "infant",
    }

    if age_str not in mapping:
        raise ValueError(f"Unsupported age_group for HRTM: {age_group}")

    return mapping[age_str]


def _normalise_gender_subject(gender_subject):
    """Normalise gender subject input."""
    if gender_subject is None:
        return None

    gender_str = str(gender_subject).strip().lower()

    if gender_str in {"male", "m"}:
        return "male"
    if gender_str in {"female", "f"}:
        return "female"

    raise ValueError(f"Unsupported gender_subject: {gender_subject}")


def _normalise_breathing_mode(breathing_mode):
    """Normalise breathing mode labels."""
    if breathing_mode is None:
        return "nasal"

    mode_str = str(breathing_mode).strip().lower()

    if mode_str in {"nasal", "nose", "normal"}:
        return "nasal"
    if mode_str in {"oral", "mouth"}:
        return "mouth"

    raise ValueError(f"Unsupported breathing_mode: {breathing_mode}")


# -----------------------------------------------------------------
# INPUT VALIDATION
# -----------------------------------------------------------------
def _validate_positive_scalar(value, name):
    """Validate positive scalar input."""
    if value is None:
        raise ValueError(f"{name} cannot be None.")
    if float(value) <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_nonnegative_array(arr, name):
    """Validate non-negative array input."""
    array = np.asarray(arr, dtype=float)
    if np.any(array < 0):
        raise ValueError(f"{name} cannot contain negative values.")
    return array


# -----------------------------------------------------------------
# ACTIVITY NAME MAPPING
# -----------------------------------------------------------------
# User-facing activity labels -> deposition module activity labels
# -----------------------------------------------------------------
DEFAULT_ACTIVITY_MAP = {
    "rest": "sleep",
    "light": "sitting",
    "moderate": "light_exercise",
    "heavy": "heavy_exercise",
}

# User-facing labels for plots
ACTIVITY_LABELS = {
    "rest": "Rest (sleep)",
    "light": "Light (sitting)",
    "moderate": "Moderate (light exercise)",
    "heavy": "Heavy (exercise)",
}


# -----------------------------------------------------------------
# BREATHING RATE EXTRACTION FROM JSON
# -----------------------------------------------------------------
def get_activity_breathing_rates(age_group="adult", gender_subject="male", breathing_mode="nasal"):
    """
    Return activity-based breathing rates [m^3/h] from JSON through
    func_hrtm_deposition.py.

    Activities unavailable in the JSON are skipped, never invented.
    """
    breathing_rates = {}

    for user_activity in ["rest", "light", "moderate", "heavy"]:
        try:
            params = get_breathing_params(
                age_group=_normalise_age_group(age_group),
                activity_level=DEFAULT_ACTIVITY_MAP[user_activity],
                gender=_normalise_gender_subject(gender_subject),
                breathing_mode=_normalise_breathing_mode(breathing_mode),
                json_path=HRTM_SUBJECTS_JSON_PATH,
            )
            breathing_rates[user_activity] = float(params["B_m3_h"])
        except ValueError:
            # Explicitly unavailable activities in the JSON are skipped.
            continue

    if len(breathing_rates) == 0:
        raise ValueError(
            f"No valid breathing activities found for age_group='{age_group}', "
            f"gender='{gender_subject}', breathing_mode='{breathing_mode}'."
        )

    return breathing_rates


# -----------------------------------------------------------------
# INTAKE AND DEPOSITION MAP CALCULATIONS
# -----------------------------------------------------------------
def compute_intake_map_bq(concentration_map_bq_m3, exposure_time_h=1.0, breathing_rate_m3_h=1.2):
    """
    Compute inhaled activity map from air concentration map.

    intake_map [Bq] = concentration_map [Bq/m^3] * breathing_rate [m^3/h] * exposure_time [h]
    """
    concentration_map = _validate_nonnegative_array(concentration_map_bq_m3, "Concentration map")
    _validate_positive_scalar(exposure_time_h, "Exposure time")
    _validate_positive_scalar(breathing_rate_m3_h, "Breathing rate")

    return concentration_map * float(exposure_time_h) * float(breathing_rate_m3_h)


def compute_deposited_activity_map(intake_map_bq, deposition_fractions):
    """
    Compute deposited activity maps by HRTM region.
    """
    intake_map = _validate_nonnegative_array(intake_map_bq, "Intake map")

    deposited_maps = {
        region: intake_map * float(deposition_fractions[region])
        for region in ["ET1", "ET2", "BB", "bb", "AI", "EXH"]
    }

    deposited_maps["extrathoracic_total"] = (
        deposited_maps["ET1"] + deposited_maps["ET2"]
    )
    deposited_maps["thoracic_total"] = (
        deposited_maps["BB"] + deposited_maps["bb"] + deposited_maps["AI"]
    )
    deposited_maps["respiratory_total"] = (
        deposited_maps["extrathoracic_total"] + deposited_maps["thoracic_total"]
    )

    return deposited_maps


# -----------------------------------------------------------------
# SINGLE-ACTIVITY HRTM RUN
# -----------------------------------------------------------------
def run_hrtm_map(
    concentration_map_bq_m3,
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
    Run HRTM over a full concentration map for a single activity case.

    If breathing_rate_m3_h is None and activity_label is provided,
    the breathing rate is taken from the JSON through
    func_hrtm_deposition.py.
    """
    concentration_map = _validate_nonnegative_array(concentration_map_bq_m3, "Concentration map")

    _validate_positive_scalar(exposure_time_h, "Exposure time")
    _validate_positive_scalar(particle_shape_factor, "Particle shape factor")
    _validate_positive_scalar(particle_density_g_cm3, "Particle density")

    age_key = _normalise_age_group(age_group)
    gender_key = _normalise_gender_subject(gender_subject)
    breathing_mode_key = _normalise_breathing_mode(breathing_mode)
    amad_key = _normalise_amad(amad)

    breathing_params = None
    fractions = None

    if activity_label is not None:
        activity_key = str(activity_label).strip().lower()
        if activity_key not in DEFAULT_ACTIVITY_MAP:
            raise ValueError(
                f"Unsupported activity_label '{activity_label}'. "
                f"Use one of: {list(DEFAULT_ACTIVITY_MAP.keys())}"
            )

        deposition_activity = DEFAULT_ACTIVITY_MAP[activity_key]

        breathing_params = get_breathing_params(
            age_group=age_key,
            activity_level=deposition_activity,
            gender=gender_key,
            breathing_mode=breathing_mode_key,
            json_path=HRTM_SUBJECTS_JSON_PATH,
        )

        fractions = get_regional_deposition_fractions(
            age_group=age_key,
            activity_level=deposition_activity,
            amad=amad_key,
            gender=gender_key,
            breathing_mode=breathing_mode_key,
            particle_density=particle_density_g_cm3,
            particle_shape_factor=particle_shape_factor,
            use_regional_sf=use_regional_sf,
            json_path=HRTM_SUBJECTS_JSON_PATH,
        )

        if breathing_rate_m3_h is None:
            breathing_rate_m3_h = float(breathing_params["B_m3_h"])

    if breathing_rate_m3_h is None:
        raise ValueError(
            "breathing_rate_m3_h cannot be None when activity_label is not provided."
        )

    _validate_positive_scalar(breathing_rate_m3_h, "Breathing rate")

    if fractions is None:
        raise ValueError(
            "Regional deposition fractions could not be resolved. "
            "Use activity_label so the subject/scenario can be taken from the JSON."
        )

    intake_map_bq = compute_intake_map_bq(
        concentration_map_bq_m3=concentration_map,
        exposure_time_h=exposure_time_h,
        breathing_rate_m3_h=breathing_rate_m3_h,
    )

    deposited_maps_bq = compute_deposited_activity_map(intake_map_bq, fractions)

    respiratory_fraction = (
        float(fractions["ET1"])
        + float(fractions["ET2"])
        + float(fractions["BB"])
        + float(fractions["bb"])
        + float(fractions["AI"])
    )

    max_idx = np.unravel_index(np.argmax(concentration_map), concentration_map.shape)

    return {
        "mode": "map",
        "shape": concentration_map.shape,
        "amad": amad_key,
        "exposure_time_h": float(exposure_time_h),
        "breathing_rate_m3_h": float(breathing_rate_m3_h),
        "activity_label": activity_label,
        "age_group_input": age_group,
        "gender_subject_input": gender_subject,
        "age_group_resolved": age_key,
        "gender_resolved": gender_key,
        "breathing_mode": breathing_mode_key,
        "fn_nasal_fraction": float(fractions["Fn"]),
        "breathing_parameters": breathing_params,
        "particle_shape_factor": float(particle_shape_factor),
        "particle_density_g_cm3": float(particle_density_g_cm3),
        "subject_anatomy": fractions["anatomy"].copy(),
        "deposition_fractions": {
            "ET1": float(fractions["ET1"]),
            "ET2": float(fractions["ET2"]),
            "BB": float(fractions["BB"]),
            "bb": float(fractions["bb"]),
            "AI": float(fractions["AI"]),
            "EXH": float(fractions["EXH"]),
        },
        "respiratory_fraction": float(respiratory_fraction),
        "concentration_map_bq_m3": concentration_map,
        "intake_map_bq": intake_map_bq,
        "deposited_activity_maps_bq": deposited_maps_bq,
        "max_concentration_bq_m3": float(np.max(concentration_map)),
        "max_intake_bq": float(np.max(intake_map_bq)),
        "max_respiratory_total_bq": float(np.max(deposited_maps_bq["respiratory_total"])),
        "max_index_rc": (int(max_idx[0]), int(max_idx[1])),
        "scenario_resolved": fractions["scenario"],
        "aero_modifier": float(fractions["aero_modifier"]),
        "use_regional_sf": bool(fractions["use_regional_sf"]),
    }


# -----------------------------------------------------------------
# ALL-ACTIVITY HRTM RUN
# -----------------------------------------------------------------
def run_hrtm_map_all_activities(
    concentration_map_bq_m3,
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
    Run HRTM over a full concentration map for all valid activity cases.
    """
    concentration_map = _validate_nonnegative_array(concentration_map_bq_m3, "Concentration map")
    _validate_positive_scalar(exposure_time_h, "Exposure time")

    if breathing_rates_m3_h is None:
        breathing_rates = get_activity_breathing_rates(
            age_group=age_group,
            gender_subject=gender_subject,
            breathing_mode=breathing_mode,
        )
    else:
        breathing_rates = {}
        for activity, br in breathing_rates_m3_h.items():
            _validate_positive_scalar(br, f"Breathing rate for activity '{activity}'")
            breathing_rates[str(activity).strip().lower()] = float(br)

    results = {}
    for activity, br in breathing_rates.items():
        results[activity] = run_hrtm_map(
            concentration_map_bq_m3=concentration_map,
            amad=amad,
            exposure_time_h=exposure_time_h,
            breathing_rate_m3_h=br,
            age_group=age_group,
            gender_subject=gender_subject,
            breathing_mode=breathing_mode,
            particle_shape_factor=particle_shape_factor,
            particle_density_g_cm3=particle_density_g_cm3,
            activity_label=activity,
            use_regional_sf=use_regional_sf,
        )

    return {
        "mode": "map_all_activities",
        "amad": _normalise_amad(amad),
        "exposure_time_h": float(exposure_time_h),
        "age_group_input": age_group,
        "gender_subject_input": gender_subject,
        "breathing_mode": _normalise_breathing_mode(breathing_mode),
        "particle_shape_factor": float(particle_shape_factor),
        "particle_density_g_cm3": float(particle_density_g_cm3),
        "use_regional_sf": bool(use_regional_sf),
        "activities": list(breathing_rates.keys()),
        "breathing_rates_m3_h": breathing_rates,
        "results_by_activity": results,
    }


# -----------------------------------------------------------------
# ACTIVITY COMPARISON PLOT
# -----------------------------------------------------------------
def save_hrtm_activity_comparison_plot(
    hrtm_results_all_activities,
    run_dir,
    filename="hrtm_activity_comparison.png",
    quantity_key="respiratory_total"
):
    """
    Save a 2x2 comparison figure showing one selected HRTM map quantity
    for all activity cases.
    """
    if hrtm_results_all_activities.get("mode") != "map_all_activities":
        raise ValueError("save_hrtm_activity_comparison_plot expects map_all_activities results.")

    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)

    results_by_activity = hrtm_results_all_activities["results_by_activity"]
    activities = list(results_by_activity.keys())

    if len(activities) == 0:
        raise ValueError("No activity results available to plot.")

    X = hrtm_results_all_activities.get("X_m", None)
    Y = hrtm_results_all_activities.get("Y_m", None)

    use_real_coordinates = X is not None and Y is not None
    if use_real_coordinates:
        extent = [np.nanmin(X), np.nanmax(X), np.nanmin(Y), np.nanmax(Y)]

    if quantity_key == "intake":
        sample_map = results_by_activity[activities[0]]["intake_map_bq"]
        title_label = "Inhaled activity"
        cbar_label = "Activity (µBq)"
    else:
        deposited_maps = results_by_activity[activities[0]]["deposited_activity_maps_bq"]
        if quantity_key not in deposited_maps:
            raise ValueError(f"Unsupported quantity_key: {quantity_key}")
        sample_map = deposited_maps[quantity_key]
        title_label = quantity_key.replace("_", " ").capitalize()
        cbar_label = "Deposited activity (µBq)"

    vmin = 0.0
    vmax = float(np.max(sample_map))
    for activity in activities[1:]:
        if quantity_key == "intake":
            current_map = results_by_activity[activity]["intake_map_bq"]
        else:
            current_map = results_by_activity[activity]["deposited_activity_maps_bq"][quantity_key]
        vmax = max(vmax, float(np.max(current_map)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(activities):
            activity = activities[i]
            result = results_by_activity[activity]

            if quantity_key == "intake":
                data_map = result["intake_map_bq"] * 1e6
            else:
                data_map = result["deposited_activity_maps_bq"][quantity_key] * 1e6

            if use_real_coordinates:
                im = ax.imshow(
                    data_map,
                    origin="lower",
                    extent=extent,
                    aspect="equal",
                    vmin=vmin * 1e6,
                    vmax=vmax * 1e6
                )
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)")
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
                ax.plot(0, 0, "ro", markersize=5)
            else:
                im = ax.imshow(
                    data_map,
                    origin="lower",
                    aspect="auto",
                    vmin=vmin * 1e6,
                    vmax=vmax * 1e6
                )
                ax.set_xlabel("Grid index X")
                ax.set_ylabel("Grid index Y")

            br = result["breathing_rate_m3_h"]
            activity_label = ACTIVITY_LABELS.get(activity, activity)
            ax.set_title(f"{activity_label} | BR = {br:.2f} m³/h")
            ax.grid(True, alpha=0.30)

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(cbar_label)
        else:
            ax.axis("off")

    amad = hrtm_results_all_activities.get("amad", "N/A")
    exposure_time_h = hrtm_results_all_activities.get("exposure_time_h", None)

    if exposure_time_h is not None:
        fig.suptitle(
            f"HRTM deposition comparison by activity | Quantity = {title_label} | AMAD = {amad} | t = {exposure_time_h:.2f} h",
            fontsize=13,
        )
    else:
        fig.suptitle(
            f"HRTM deposition comparison by activity | Quantity = {title_label} | AMAD = {amad}",
            fontsize=13,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(run_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path


# -----------------------------------------------------------------
# REGIONAL ACTIVITY BAR PLOT
# -----------------------------------------------------------------
def save_hrtm_regional_activity_barplot(
    hrtm_results_all_activities,
    run_dir,
    filename="hrtm_regional_activity_barplot.png"
):
    """
    Save a single grouped bar-chart figure comparing deposited activity by
    respiratory region at the maximum respiratory-deposition point for each
    activity case [mBq].
    """
    if hrtm_results_all_activities.get("mode") != "map_all_activities":
        raise ValueError("save_hrtm_regional_activity_barplot expects map_all_activities results.")

    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)

    results_by_activity = hrtm_results_all_activities["results_by_activity"]
    activities = list(results_by_activity.keys())

    if len(activities) == 0:
        raise ValueError("No activity results available to plot.")

    regions = ["ET1", "ET2", "BB", "bb", "AI"]
    activity_labels = [ACTIVITY_LABELS.get(activity, activity) for activity in activities]

    activity_colors = {
        "rest": "#2166AC",
        "light": "#67A9CF",
        "moderate": "#F46D43",
        "heavy": "#D73027",
    }

    deposited_values_millibq = {activity: [] for activity in activities}

    for activity in activities:
        result = results_by_activity[activity]
        deposited_maps = result["deposited_activity_maps_bq"]

        idx_resp_max = np.unravel_index(
            np.nanargmax(deposited_maps["respiratory_total"]),
            deposited_maps["respiratory_total"].shape
        )

        for region in regions:
            deposited_values_millibq[activity].append(
                float(deposited_maps[region][idx_resp_max]) * 1e3
            )

    x = np.arange(len(regions))
    width = 0.18

    fig, ax = plt.subplots(figsize=(14, 6.5))

    for i, activity in enumerate(activities):
        offset = (i - (len(activities) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            deposited_values_millibq[activity],
            width=width,
            label=activity_labels[i],
            color=activity_colors.get(activity, "#70AD47")
        )

        for bar, value in zip(bars, deposited_values_millibq[activity]):
            y_text = bar.get_height() * 0.5
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_text,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                rotation=90,
                color="white",
                fontweight="bold"
            )

    ax.set_title("HRTM regional deposited activity at maximum deposition point")
    ax.set_ylabel("Deposited activity (mBq)")
    ax.set_xlabel("Respiratory region")
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Activity")

    breathing = hrtm_results_all_activities.get("breathing_mode", "N/A")
    amad = hrtm_results_all_activities.get("amad", "N/A")
    exposure_time_h = hrtm_results_all_activities.get("exposure_time_h", None)

    if exposure_time_h is not None:
        fig.suptitle(
            f"HRTM regional comparison by activity | {breathing} | AMAD = {amad} (μm) | t = {exposure_time_h:.2f} h",
            fontsize=13,
        )
    else:
        fig.suptitle(
            f"HRTM regional comparison by activity | {breathing} | AMAD = {amad} (μm)",
            fontsize=13,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(run_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path