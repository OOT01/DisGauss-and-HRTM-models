import pandas as pd

from dgm_hrtm import run_simulations_from_dataframe

print("Starting batch test...")

df_inputs = pd.DataFrame([
    {
        "max_x": 1000.0,
        "max_y": 1000.0,
        "grid_resolution": 100,
        "evaluation_height_z": 2.0,
        "emission_rate_Q": 1.0e6,
        "emission_height_H": 10.0,
        "wind_components": [0.0, 0.0],
        "dispersion_model": "Briggs_r",
        "stability_index": 3,
        "colormap_index": 0,
        "radionuclide": "I-131",
        "population": "worker",
        "exposure_pathway": "inhalation",
        "absorption_type": "F",
        "age_group": "adult",
        "amad": "5um",
        "ingestion_key": None,
        "gender_subject": "male",
        "breathing_mode": "nasal",
        "particle_shape_factor": 1.5,
        "particle_density_g_cm3": 3.0,
        "coordinates": [37.39, -5.99],
        "meteo_mode": "current",
        "meteo_date": None,
        "meteo_hour": None,
        "exposure_time_h": 1.0,
        "use_regional_sf": False,
        "use_mapbox": False,
    }
])

print("Input DataFrame created.")
print("Launching simulation...")

batch_results = run_simulations_from_dataframe(df_inputs)

print("Simulation finished.")
print(f"Number of simulations run: {len(batch_results)}")

results = batch_results[0]

print("Results directory:")
print(results.results_dir)

print("\nReturned DataFrames:")
print(list(results.dataframes.keys()))

print("\nReturned Arrays:")
print(list(results.arrays.keys()))

print("\nReturned Figures:")
print(list(results.figures.keys()))

print("\nReturned Metadata keys:")
print(list(results.metadata.keys()))