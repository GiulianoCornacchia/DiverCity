<div align="center">
  <h1>A universal framework for route diversification in road networks</h1> 
</div>

### Authors:

* Giuliano Cornacchia <sup>1,2</sup> [<img src="https://img.shields.io/badge/ORCID-0000--0003--2263--7654-brightgreen?logo=orcid&logoColor=white" alt="ORCID" height="16">](https://orcid.org/0000-0003-2263-7654)

* Luca Pappalardo <sup>1,3</sup> [<img src="https://img.shields.io/badge/ORCID-0000--0002--1547--6007-brightgreen?logo=orcid&logoColor=white" alt="ORCID" height="16">](https://orcid.org/0000-0002-1547-6007)

* Mirco Nanni <sup>1</sup> [<img src="https://img.shields.io/badge/ORCID-0000--0003--3534--4332-brightgreen?logo=orcid&logoColor=white" alt="ORCID" height="16">](https://orcid.org/0000-0003-3534-4332)

* Dino Pedreschi <sup>2</sup> [<img src="https://img.shields.io/badge/ORCID-0000--0003--4801--3225-brightgreen?logo=orcid&logoColor=white" alt="ORCID" height="16">](https://orcid.org/0000-0003-4801-3225)

* Marta C. Gonzalez <sup>4,5,6</sup> [<img src="https://img.shields.io/badge/ORCID--0000--0002--8482--0318-brightgreen?logo=orcid&logoColor=white" alt="ORCID" height="16">](https://orcid.org/0000-0002-8482-0318)




Affiliations:<br>
<sup>1</sup> Institute of Information Science and Technologies (ISTI), National Research Council (CNR), Pisa, Italy <br>
<sup>2</sup> Department of Computer Science, University of Pisa, Pisa, Italy <br>
<sup>3</sup> Scuola Normale Superiore, Pisa, Italy <br>
<sup>4</sup> Department of City and Regional Planning, University of California, Berkeley, CA, USA <br>
<sup>5</sup> Energy Technologies Area, Lawrence Berkeley National Laboratory, Berkeley, CA, USA <br>
<sup>6</sup> Department of Civil and Environmental Engineering, University of California, Berkeley, CA, USA <br>


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
## Abstract

The structure of road networks significantly impacts various urban dynamics, from traffic congestion to environmental sustainability and equitable access to services. 
Recent studies reveal that most roads are underutilized, faster alternative routes are often overlooked, and traffic is typically concentrated on a few corridors.
In this article, we examine how road network topology, and in particular the presence of mobility attractors (e.g., highways and ring roads), shapes the counterpart to traffic concentration: route diversification. 
To this end, we introduce DiverCity, a measure that quantifies the extent to which traffic can potentially be distributed across multiple, loosely overlapping routes.  
Analyzing 56 global cities with diverse population densities and road network topologies, we find that DiverCity is closely tied to traffic efficiency and network characteristics such as network extensiveness and number of intersections. 
Within cities, DiverCity increases with distance from the city center before stabilizing in the periphery but declines in the proximity of mobility attractors.
We demonstrate that strategic speed limit adjustments on mobility attractors can increase DiverCity while preserving travel efficiency.
We isolate the complex interplay between mobility attractors and DiverCity through simulations in a controlled setting, confirming the patterns observed in real-world cities. 
DiverCity provides a practical tool for urban planners and policymakers to optimize road network design and balance route diversification, efficiency, and sustainability.
