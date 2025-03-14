<div align="center">
  <h1>A universal framework for route diversification <br> in road networks</h1> 
</div>

### Authors:

* Giuliano Cornacchia <sup>1,2</sup> [<img src="https://img.shields.io/badge/ORCID-0000--0003--2263--7654-brightgreen?logo=orcid&logoColor=white" alt="ORCID" height="16">](https://orcid.org/0000-0003-2263-7654)

* Luca Pappalardo <sup>1,3</sup> [<img src="https://img.shields.io/badge/ORCID-0000--0002--1547--6007-brightgreen?logo=orcid&logoColor=white" alt="ORCID" height="16">](https://orcid.org/0000-0002-1547-6007)

* Mirco Nanni <sup>1</sup> [<img src="https://img.shields.io/badge/ORCID-0000--0003--3534--4332-brightgreen?logo=orcid&logoColor=white" alt="ORCID" height="16">](https://orcid.org/0000-0003-3534-4332)

* Dino Pedreschi <sup>2</sup> [<img src="https://img.shields.io/badge/ORCID-0000--0003--4801--3225-brightgreen?logo=orcid&logoColor=white" alt="ORCID" height="16">](https://orcid.org/0000-0003-4801-3225)

* Marta C. Gonzalez <sup>4,5,6</sup> [<img src="https://img.shields.io/badge/ORCID--0000--0002--8482--0318-brightgreen?logo=orcid&logoColor=white" alt="ORCID" height="16">](https://orcid.org/0000-0002-8482-0318)

<p style="font-size: xx-small;">
Affiliations:<br>
<sup>1</sup> Institute of Information Science and Technologies (ISTI), National Research Council (CNR), Pisa, Italy <br>
<sup>2</sup> Department of Computer Science, University of Pisa, Pisa, Italy <br>
<sup>3</sup> Scuola Normale Superiore, Pisa, Italy <br>
<sup>4</sup> Department of City and Regional Planning, University of California, Berkeley, CA, USA <br>
<sup>5</sup> Energy Technologies Area, Lawrence Berkeley National Laboratory, Berkeley, CA, USA <br>
<sup>6</sup> Department of Civil and Environmental Engineering, University of California, Berkeley, CA, USA <br>
</p>

____

Pre-print coming soon
____


## Built with

![python](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white)
![jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)
![numpy](https://img.shields.io/badge/NumPy-013243.svg?style=for-the-badge&logo=NumPy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![osm](https://img.shields.io/badge/OpenStreetMap-7EBC6F.svg?style=for-the-badge&logo=OpenStreetMap&logoColor=white)

### Requirements

This project uses the following versions:

![Python](https://img.shields.io/badge/Python-3.9.18-blue)

- **Python**: The code is written in Python 3.9.18.

<a id='abstract' name='abstract'></a>
## Overview

The structure of road networks significantly impacts urban dynamics, influencing traffic congestion, environmental sustainability, and equitable access to services. DiverCity quantifies the potential for distributing traffic across multiple, loosely overlapping routes, revealing how road network topology and mobility attractors (e.g., highways and ring roads) influence route diversification. Analyzing 56 global cities, this study shows that DiverCity is linked to traffic efficiency and network characteristics such as extensiveness and number of intersections. Additionally, strategic speed limit adjustments on mobility attractors can increase DiverCity while preserving travel efficiency.

An interactive platform is available to explore the spatial distribution of DiverCity in the cities analyzed: https://divercitymaps.github.io


## The DiverCity Measure in a Nutshell

### How Many Ways Can You Get There?

How many ways can you get from home to your favorite café? And how different are those routes from each other?


**DiverCity** answers this by measuring:
- **How many practical alternative routes exist between two points?**
- **How different are those routes from each other?**

It analyzes **Near-Shortest Routes (NSRs)**—paths that are only slightly longer than the fastest route—by focusing on:
1. **Number of Alternatives**: How many NSRs are available.
2. **Spatial Spread**: How diverse these routes are, calculated using **Weighted Jaccard Similarity** to measure overlap.

The DiverCity of a trip from home to the café is calculated as:

$D(\text{home}, \text{café}) = S(NSR(\text{home}, \text{cafè})) \cdot |NSR(\text{home}, \text{cafè})|$

Where:
- $NSR(\text{home}, \text{cafè})$ is the set of near-shortest routes.
- $S(NSR(\text{home}, \text{cafè}))$ is the spatial spread of these routes.

$S(NSR) = 1 - J(NSR)$, where $J$ is the average pair-wise weighted Jaccard similarity among the NSR.

---

### Why It Matters

DiverCity shows how well a city’s road network can distribute traffic across multiple routes. It helps:
- **Urban Planners** design more balanced traffic flows.
- **Policymakers** assess the impact of interventions like speed limit adjustments.
- **Researchers** explore how network structures influence urban mobility.

By quantifying route diversification, DiverCity identifies areas within a city that are more prone to congestion due to limited alternative routes.

---

### Real-World Insights

- **High Diversification**: Grid-structured cities (e.g., Chicago, New York) offer more diverse route options.
- **Low Diversification**: Cities like **Mumbai** and **Rome** concentrate traffic through fewer corridors.
- **Mobility Attractors Impact**: DiverCity decreases near highways and ring roads, which channel traffic through fast corridors, reducing route diversity nearby.

---

### Explore DiverCity

Discover route diversification in 56 global cities using our **interactive platform**. Choose any two points to:
- Compute alternative routes and measure their DiverCity.
- Visualize route diversification as a heatmap.

Try it out: [https://divercitymaps.github.io](https://divercitymaps.github.io)


---

## Code Descriptions

The repository consists of several Jupyter Notebooks and Python scripts designed to compute and analyze DiverCity metrics for urban road networks. Below is a detailed description of each component.

---

### Notebooks

- **`1_Download_Road_Network.ipynb`**:  
  This notebook is responsible for downloading the road network data of a city using OpenStreetMap (OSMnx). It saves the road network in a compressed GraphML format for efficient storage and loading.

- **`2_Compute_DiverCity.ipynb`**:  
  This notebook computes DiverCity metrics, which quantify route diversification by analyzing alternative routes between origin-destination pairs within a city. It leverages radial sampling to gnere origin-destination pairs and path penalization to generate alternative routes. It computes DiverCity for each trip considering number and the spatial spread of near-shortest routes.

---

### Scripts

- **`compute_divercity_osm.py`**:
This is the main script for computing DiverCity metrics using the radial sampling method and path penalization technique. It loads or downloads the road network, performs radial sampling, computes alternative routes, and calculates DiverCity metrics. Results are saved as JSON files for further analysis and visualization.

---

#### Parameters Table for `compute_divercity_osm.py`

| Parameter           | Description                                                     | Required | Default Value        |
|---------------------|-----------------------------------------------------------------|----------|----------------------|
| `-c`, `--city`      | City name used to load the corresponding road network             | Yes      | None                 |
| `-p`, `--plist`     | List of penalization factors for Path Penalization (PP)           | Yes      | None                 |
| `-e`, `--epslist`   | List of epsilon values for identifying Near-Shortest Routes (NSR) | Yes      | None                 |
| `--lat`             | Latitude of the city center                                       | Yes      | None                 |
| `--lng`             | Longitude of the city center                                      | Yes      | None                 |
| `-i`, `--identifier`| Experiment identifier for result storage                         | Yes      | None                 |
| `-a`, `--attribute` | Path attribute for the shortest path (e.g., 'traveltime')         | No       | `traveltime`          |
| `-f`, `--rfrom`     | Starting radius (km) for radial sampling                         | No       | 1                    |
| `-t`, `--rto`       | Ending radius (km) for radial sampling                           | No       | 30                   |
| `-s`, `--rstep`     | Radius step (km) for concentric circles in radial sampling        | No       | 1                    |
| `-k`                | Number of alternative routes to consider                         | No       | 10                   |
| `-n`, `--ncircles`  | Number of samples per circle for radial sampling                 | No       | 36                   |
| `-r`, `--saveroutes`| Save generated routes (1 = Yes, 0 = No)                          | No       | 0                    |
| `-l`, `--reducespeed`| Speed reduction factor for mobility attractors                  | No       | 1                    |
| `--njobs`           | Number of parallel jobs for computation                          | No       | 5                    |

Example Command:
```bash
python compute_divercity_osm.py -c Rome -p "[0.1,0.2,0.3]" -e "[0.1,0.2,0.3]" --lat 41.9028 --lng 12.4964 -i rome_test


