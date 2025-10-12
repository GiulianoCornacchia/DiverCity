<div align="center">
  <h1>A computational framework for quantifying route diversification in road networks</h1> 
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

Pre-print: https://www.arxiv.org/abs/2510.02582

If you use the code in this repository, please cite our paper:

```
@misc{cornacchia2025divercity,
      title={A computational framework for quantifying route diversification in road networks}, 
      author={Giuliano Cornacchia and Luca Pappalardo and Mirco Nanni and Dino Pedreschi and Marta C. González},
      year={2025},
      eprint={2510.02582},
      archivePrefix={arXiv},
      primaryClass={physics.soc-ph},
      url={https://arxiv.org/abs/2510.02582}, 
}
```


____


## Built with

![python](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white)
![jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)
![numpy](https://img.shields.io/badge/NumPy-013243.svg?style=for-the-badge&logo=NumPy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![osm](https://img.shields.io/badge/OpenStreetMap-7EBC6F.svg?style=for-the-badge&logo=OpenStreetMap&logoColor=white)

### Requirements

This project uses the following versions:

![Python](https://img.shields.io/badge/Python-3.10-blue)

* **Python**: 3.10 (recommended). The code uses OSMnx + igraph and parallel processing.
* `environment.yml` is provided for reproducibility. Install via `conda env create -f environment.yml`.

<a id='abstract' name='abstract'></a>
## Overview

The structure of road networks significantly impacts urban dynamics, influencing traffic congestion, environmental sustainability, and equitable access to services. DiverCity quantifies the potential for distributing traffic across multiple, loosely overlapping routes, revealing how road network topology and mobility attractors (e.g., highways and ring roads) influence route diversification. Analyzing 56 global cities, this study shows that DiverCity is linked to traffic efficiency and network characteristics such as extensiveness and number of intersections. Additionally, strategic speed limit adjustments on mobility attractors can increase DiverCity while preserving travel efficiency.

An interactive platform is available to explore the spatial distribution of DiverCity in the cities analyzed: https://divercitymaps.github.io


## The DiverCity Measure in a Nutshell

### How Many Ways Can You Get There?

How many different ways can you get from home to your favorite café? And how different are those routes from one another?

**DiverCity** measures both the *number* and the *diversity* of practical route alternatives between two points in a road network. It answers two main questions:

* **How many near-shortest routes exist between two points?**
* **How spatially different are these routes from each other?**

It focuses on **Near-Shortest Routes (NSRs)**, paths that are only slightly longer than the fastest one, by combining:

1. **Number of Alternatives**: The count of NSRs found between an origin and a destination.
2. **Spatial Spread**: How different those NSRs are, quantified using the **Weighted Jaccard Similarity** between their edges.

Formally, the DiverCity of a trip from *home* to *café* is defined as:

$D(\text{home}, \text{café}) = S(NSR(\text{home}, \text{café})) \cdot |NSR(\text{home}, \text{café})|$

Where:

* $NSR(\text{home}, \text{café})$ is the set of near-shortest routes.
* $|NSR|$ is the number of routes in that set.
* $S(NSR) = 1 - J(NSR)$ measures the spatial spread of the routes.
* $J(NSR)$ is the average pairwise weighted Jaccard similarity among all NSRs, representing their overlap.

A high DiverCity means that multiple distinct routes exist with comparable travel times.
A low DiverCity means that all practical routes are nearly identical, following the same main roads.

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

* **`1_Download_Road_Network.ipynb`**:
  This notebook downloads the road network of a selected city using **OpenStreetMap** data via **OSMnx**. The user specifies the city name and coordinates, and the notebook extracts the `drive` network within a chosen radius. The resulting graph is enriched with edge attributes such as length, speed, and travel time, and is then saved as a compressed **GraphML** file for later use.

* **`2_Compute_DiverCity.ipynb`**:
  This notebook performs the computation of **DiverCity** metrics for a given city. It first generates origin-destination pairs through **radial sampling** around the city center, ensuring spatial coverage across multiple distances. For each pair, it uses the **path penalization** algorithm to generate up to *k* alternative near-shortest routes and calculates DiverCity by combining their number and spatial spread.

* **`3_Attractors_Measures.ipynb`**:
  This notebook analyzes **mobility attractors**, defined as high-capacity infrastructures such as highways, ring roads, and major arterial roads. It computes attractor-related measures, including total length, spatial density, and spatial dispersion, by sampling random points and measuring their distance to the nearest attractor. The outputs help quantify how the configuration of attractors influences route diversification.

* **`4_Simplified_Model.ipynb`**:
  This notebook introduces a **synthetic grid-based model** to test DiverCity mechanisms in a controlled environment. It creates artificial networks with configurable parameters such as attractor corridors, water bodies, and bridges, and then simulates speed adjustments and attractor placement. The results confirm how network structure and speed limits affect route diversification independently of real-world complexities.

* **`5a_Plot_Routes.ipynb`**:
  This notebook visualizes the alternative routes generated for selected origin-destination pairs. It highlights **near-shortest routes (NSR)** and compares them with longer alternatives to show how route overlap and diversity vary across cities and scenarios. The visualizations help interpret DiverCity values in geographic terms.

* **`5b_Maps_DiverCity.ipynb`**:
  This notebook creates **spatial maps** of DiverCity within a city. It computes node-level DiverCity by aggregating trip-level measures and interpolates the values across the urban area. The resulting heatmaps illustrate how route diversification changes with distance from the city center and near mobility attractors.

* **`6_Figures.ipynb`**:
  This notebook reproduces the **main figures** of the DiverCity study. It uses processed results to generate plots such as DiverCity distributions across cities, DiverCity versus radial distance, and DiverCity improvements under speed-limit adjustments. Each figure corresponds to those presented in the paper and supports visual validation of the study’s findings.

---

### (most relevant) Scripts

* **`compute_divercity_osm.py`**:
  This is the **main execution script** for computing DiverCity metrics across cities. It loads or downloads the corresponding road network, performs **radial sampling** of origin-destination pairs, and applies the **path penalization** algorithm to generate multiple near-shortest routes. The script computes DiverCity values for each trip, combining the number and spatial diversity of the obtained routes.

* **`divercity_utils.py`**:
  This module contains the **core functions** for DiverCity computation. It implements the logic to evaluate near-shortest routes, calculate weighted Jaccard similarity, derive spatial spread, and aggregate DiverCity values at the trip and city levels.

* **`routing_utils.py`**:
  This module provides **routing and graph-processing utilities**. It handles shortest-path calculations, iterative edge penalization, and parallelized path generation using NetworkX and igraph.

* **`simplified_model.py`**:
  This module implements the **synthetic grid-based model** used to study DiverCity mechanisms in an abstract, controlled setting. It allows users to define grid size, attractor placement, water obstacles, and bridge connections, and to simulate changes in speed limits or network configuration.

*Other scripts* such as `figures.py`, `network_measures.py`, `results_utils.py`, `route_plotting.py`, and `trip_measures.py` act as **helper modules**. They handle data post-processing, visualization, and computation of supplementary measures used for figure generation, network characterization, and validation of the main results.


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


