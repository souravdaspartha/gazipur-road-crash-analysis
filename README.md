# Gazipur Road Crash Hotspot Map

This repository contains the Python workflow used to produce the **Gazipur road crash hotspot maps** presented in my [mapping portfolio](https://souravdaspartha.github.io/maps/).

The maps visualize severity-weighted crash density, accident severity, and the five largest accident-location clusters within Gazipur City Corporation.

## Map Components

The workflow produces three principal map layers:

1. **Crash Density** : a heatmap weighted by accident severity
2. **Crash Severity** : crash points classified by severity
3. **Top 5 Accident Locations** : the five largest DBSCAN clusters

The maps also display the Gazipur City Corporation boundary and major roads.

## Data Preparation

The script:

- Reads `Road_Crash_Data.csv`
- Removes the dataset summary row
- Detects and parses the date format
- Removes records with unreadable dates or missing coordinates
- Converts latitude and longitude to numeric values
- Removes coordinates outside Bangladesh
- Excludes records north of latitude 24.10
- Excludes 2019 because only two records were available for that year
- Assigns weights based on accident severity

After excluding three records outside the defined study area and two records from 2019, the final maps contain **327 crashes** recorded between **2 January 2020 and 27 August 2024**.

## Severity Weights

| Accident severity | Weight |
|---|---:|
| Fatal | 5 |
| Severe Injury | 3 |
| Minor Injury | 1 |
| Collision | 1 |

These weights determine the contribution of each recorded crash to the density heatmap.

## Hotspot Identification

Accident-location clusters are identified using DBSCAN with Haversine distance.

| Parameter | Value |
|---|---:|
| Search distance | 0.5 km |
| Minimum crashes | 5 |
| Number displayed | 5 |
| Display-circle radius | 400 metres |

The five locations are labelled as:

- Board Bazar Area
- Vogra Area
- Tongi Area
- Konabari Area
- Salna Area

## Map Design

The map uses:

- CartoDB Dark Matter basemap
- GCC road network from `gcc-road.shp`
- GCC boundary from `GCC_BND.shp`
- Severity-weighted density colours
- Separate severity symbols
- Labelled hotspot circles
- Layer controls
- Scale control
- North arrow
- Year-by-year animation

The road network is clipped to the GCC boundary before being displayed.

## Generated Outputs

The script generates:

```text
accident_hotspot_map.html
accident_hotspot_animated.html
accident_hotspot_embed.html
top_5_hotspots.csv
hotspot_clusters.csv
```

### HTML maps

- `accident_hotspot_map.html` — interactive map containing all three analytical layers
- `accident_hotspot_animated.html` — interactive year-by-year crash-density map
- `accident_hotspot_embed.html` — locked animated map prepared for embedding in the portfolio website

### Tables

- `top_5_hotspots.csv` — summary of the five largest accident-location clusters
- `hotspot_clusters.csv` — records assigned to DBSCAN clusters

## Required Local Files

The workflow uses the following local files:

```text
Road_Crash_Data.csv
gcc-road.shp
gcc-road.shx
gcc-road.dbf
gcc-road.prj
GCC_BND.shp
GCC_BND.shx
GCC_BND.dbf
GCC_BND.prj
```

These data files are not included in the public repository unless their publication is authorized.

## Python Libraries

- NumPy
- pandas
- Folium
- scikit-learn
- GeoPandas
- Shapely

## Running the Script

Place the authorized input files in one project folder and update:

```python
FOLDER = r"path\to\your\project\folder"
```

The script will load the crash, road, and boundary data and generate the HTML maps and CSV outputs in that folder.

## Data Availability

The record-level police crash data are not publicly available through this repository because of data confidentiality and usage restrictions.

## Interpretation

The maps show concentrations within the available police-recorded crash data. They are not exposure-adjusted risk maps because the analysis does not incorporate traffic volume, vehicle-kilometres travelled, or road length.

## Author

**Sourav Das Partha**  
Transport Planner and Spatial Data Analyst

- [Portfolio](https://souravdaspartha.github.io/)
- [LinkedIn](https://www.linkedin.com/in/sourav-das-partha)
- [Email](mailto:souravdaspartha@gmail.com)
