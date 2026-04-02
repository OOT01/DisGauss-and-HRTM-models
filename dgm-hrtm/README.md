# DGM-HRTM

Gaussian atmospheric dispersion model coupled with a simplified Human Respiratory Tract Model (HRTM) for radiological exposure assessment.

---

## Description

This project implements:

- A **Gaussian plume dispersion model** for atmospheric transport of radionuclides
- Integration with **meteorological data (Open-Meteo API)**
- A **Human Respiratory Tract Model (HRTM)** for inhalation dose estimation
- Support for:
  - Worker and public exposure
  - Age-dependent dose coefficients (ICRP-based)
  - Particle properties (density, shape factor)
- Optional **Mapbox-based geospatial visualization**

---

## Installation

Clone the repository and install in editable mode:

```bash
pip install dgm-hrtm
```

---

## Usage 

Run the interactive CLI:

dgm-run

You will be prompted to enter:

 - Domain parameters
 - Source/emission parameters
 - Dispersion model settings
 - Radionuclide and dose parameters
 - HRTM parameters
 - Meteorological configuration

---
## Outputs

The simulation generates:

 - 2D concentration maps
 - Dose maps (Sv)
 - HRTM deposition results
 - Figures and plots
 - parameters.txt with full configuration

All outputs are stored in a RESULTS/ directory.

---

## Optional: Mapbox Visualization

This project includes optional support for Mapbox to display real-scale geographic backgrounds in the 2D plume plots.

To use this feature, you must provide your own Mapbox access token.

### How to get a Mapbox token
  - Create an account at https://www.mapbox.com
  - Go to your account dashboard → Access Tokens
  - Click Create a token
  - Use a public token (starts with pk.)
  - Copy the generated token

### How to configure it

You can provide your token in two ways:

#### Option 1 (recommended): Environment variable

Linux / macOS:

```bash
export MAPBOX_TOKEN="your_token_here"
```

Windows PowerShell:

```powershell
$env:MAPBOX_TOKEN="your_token_here"
```

#### Option 2: Direct configuration

If your installation includes a configuration file or constant, replace:

MAPBOX_TOKEN = "YOUR_TOKEN_HERE"

with your token.

### Enable Mapbox in the simulation

Make sure the Mapbox option is enabled in your configuration:

use_mapbox = True


### Notes
 - The Mapbox feature is optional — the simulation works without it
 - Do not share your token publicly
 - If maps do not load, verify that:
    - The token is correct
    - Mapbox is enabled

---

## Authors

- Óscar Olmo Torrecillas
- Jorge Berenguer Antequera
- José Francisco Benavente Cuevas