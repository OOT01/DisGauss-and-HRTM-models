import pandas as pd

from dgm_hrtm import run_simulations_from_dataframe


def main():
    # =========================================================
    # INPUT DATAFRAME: one row = one simulation case
    # =========================================================
    df_inputs = pd.DataFrame([
        {
            "max_x": 1000.0,
            "max_y": 1000.0,
            "grid_resolution": 100,
            "evaluation_height_z": 2.0,
            "emission_rate_Q": 1.0e6,
            "emission_height_H": 10.0,
            "wind_components": [0.0, 0.0],   # overwritten later by meteo
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
        },
        {
            "max_x": 1200.0,
            "max_y": 1200.0,
            "grid_resolution": 120,
            "evaluation_height_z": 2.0,
            "emission_rate_Q": 2.0e6,
            "emission_height_H": 20.0,
            "wind_components": [0.0, 0.0],   # overwritten later by meteo
            "dispersion_model": "Briggs_u",
            "stability_index": 2,
            "colormap_index": 1,
            "radionuclide": "I-131",
            "population": "worker",
            "exposure_pathway": "inhalation",
            "absorption_type": "F",
            "age_group": "adult",
            "amad": "5um",
            "ingestion_key": None,
            "gender_subject": "female",
            "breathing_mode": "oral",
            "particle_shape_factor": 1.5,
            "particle_density_g_cm3": 3.0,
            "coordinates": [40.42, -3.70],
            "meteo_mode": "current",
            "meteo_date": None,
            "meteo_hour": None,
            "exposure_time_h": 1.0,
            "use_regional_sf": False,
            "use_mapbox": False,
        },
    ])

    print("====================================================")
    print(" DGM-HRTM BATCH API DEMONSTRATION")
    print("====================================================")
    print(f"Number of input cases: {len(df_inputs)}")

    # =========================================================
    # RUN ALL SIMULATIONS FROM THE INPUT DATAFRAME
    # =========================================================
    batch_results = run_simulations_from_dataframe(df_inputs)

    print(f"Number of completed simulations: {len(batch_results)}")

    # =========================================================
    # ACCESS RETURNED OBJECTS
    # =========================================================
    for i, results in enumerate(batch_results, start=1):
        print("\n====================================================")
        print(f" SIMULATION {i}")
        print("====================================================")

        print(f"Results directory: {results.results_dir}")

        print("\nReturned DataFrames:")
        print(list(results.dataframes.keys()))

        print("\nReturned Arrays:")
        print(list(results.arrays.keys()))

        print("\nReturned Figures:")
        print(list(results.figures.keys()))

        print("\nReturned Metadata keys:")
        print(list(results.metadata.keys()))

        # -----------------------------------------------------
        # Example 1: work with a returned DataFrame
        # -----------------------------------------------------
        if "concentration_map" in results.dataframes:
            df_conc = results.dataframes["concentration_map"]

            print("\nExample: concentration_map head")
            print(df_conc.head())

            max_conc = df_conc["concentration_bq_m3"].max()
            print(f"\nMaximum concentration from DataFrame: {max_conc:.6e} Bq/m³")

        # -----------------------------------------------------
        # Example 2: work with a returned array
        # -----------------------------------------------------
        if "concentration" in results.arrays:
            conc_array = results.arrays["concentration"]
            print(f"Maximum concentration from array: {conc_array.max():.6e} Bq/m³")

        # -----------------------------------------------------
        # Example 3: access generated image path
        # -----------------------------------------------------
        if "plume_2D_3D_combined" in results.figures:
            print("\nGenerated figure path:")
            print(results.figures["plume_2D_3D_combined"])

        # -----------------------------------------------------
        # Example 4: access metadata
        # -----------------------------------------------------
        print("\nMetadata summary:")
        print(f"Radionuclide: {results.metadata.get('radionuclide')}")
        print(f"Population: {results.metadata.get('population')}")
        print(f"Exposure pathway: {results.metadata.get('exposure_pathway')}")
        print(f"Wind magnitude: {results.metadata.get('u_total')}")
        print(f"Run directory: {results.metadata.get('run_dir')}")

    print("\n====================================================")
    print(" DEMO FINISHED SUCCESSFULLY")
    print("====================================================")


if __name__ == "__main__":
    main()