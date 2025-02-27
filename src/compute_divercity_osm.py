import os
import osmnx as ox
import numpy as np
import igraph as ig
import json
import argparse
from matplotlib import pyplot as plt
from collections import defaultdict

import multiprocessing
import pandas as pd
import gzip

import lz4.frame
import msgpack

import concurrent
from concurrent.futures import ProcessPoolExecutor, as_completed

from divercity_utils import perform_sampling, parallel_compute_divercity_score_weighted, get_attractors_by_road_types
from routing_utils import compute_path_penalization_r


# ================================
# Argument Parsing
# ================================

def parse_arguments():
    """Parse command-line arguments with detailed help descriptions based on DiverCity terminology."""
    parser = argparse.ArgumentParser(description="Compute alternative routes and DiverCity metrics.")

    # Required Arguments
    parser.add_argument('-c', '--city', required=True, help="City name. Used to load the corresponding road network if exists.")
    parser.add_argument('-p', '--plist', type=str, required=True, 
                        help="List of penalization factors for Path Penalization (PP). "
                             "This defines the level of deviation from the fastest path, influencing route diversity.")
    parser.add_argument('-e', '--epslist', type=str, required=True, 
                        help="List of epsilon values for identifying Near-Shortest Routes (NSR). "
                             "NSRs are alternative routes whose cost deviates by no more than 100*epsilon % from the fastest route.")
    parser.add_argument('--lat', type=float, required=True, help="Latitude of the city center.")
    parser.add_argument('--lng', type=float, required=True, help="Longitude of the city center.")
    parser.add_argument('-i', '--identifier', type=str, required=True, help="Experiment identifier for result storage.")

    # Optional Arguments
    parser.add_argument('-a', '--attribute', type=str, default="traveltime", 
                        help="Path attribute for the shortest path. Default is 'traveltime'.")
    parser.add_argument('-f', '--rfrom', type=int, default=1, 
                        help="Starting radius (km) for radial sampling.")
    parser.add_argument('-t', '--rto', type=int, default=30, 
                        help="Ending radius (km) for radial sampling.")
    parser.add_argument('-s', '--rstep', type=int, default=1, 
                        help="Radius step (km) for concentric circles in radial sampling.")
    parser.add_argument('-k', type=int, default=10, 
                        help="Number of alternative routes to consider.")
    parser.add_argument('-n', '--ncircles', type=int, default=36, 
                        help="Number of samples per circle for radial sampling.")
    parser.add_argument('-r', '--saveroutes', type=int, default=0, 
                        help="Save generated routes (1) or not (0). Useful for plotting, post-analysis, or debugging.")
    parser.add_argument('-l', '--reducespeed', type=float, default=1, 
                        help="Speed reduction factor for mobility attractors.")
    parser.add_argument('--njobs', type=int, default=5, 
                        help="Number of parallel jobs for computation.")
    
    return parser.parse_args()



# PARSE THE ARGUMENTS
args = parse_arguments()


# City and network parameters
city = args.city
city_center = (args.lat, args.lng)
exp_id = args.identifier
network_type = "drive"
save_routes = args.saveroutes

N_jobs = args.njobs

# Radius parameters (in km and m)
min_radius_km = args.rfrom
max_radius_km = args.rto
radius_step_km = args.rstep
radius_city_m = (max_radius_km+1) * 1000  # Buffer for the city's maximum radius in meters

# Sampling and distance parameters
th_distance_km = 0.5                  # Threshold distance
n_samples_circle = args.ncircles      # Number of samples per circle

# Path Penalization (PP) parameters
attribute = args.attribute
list_p = [float(item) for item in args.plist[1:-1].split(',')] # List of penalization factors for PP

k = args.k
k_is_route_count = True               # Whether k represents route count
max_it = 1000                         # Max iterations for the algorithm

# Divercity  parameters
list_eps = [float(item) for item in args.epslist[1:-1].split(',')] # List of epsilon values

reduce_speed_by = args.reducespeed

# output folder
if reduce_speed_by == 1:
    output_folder = f"../data/results/{city}_{exp_id}/"
else:
    output_folder = f"../data/results/{city}_speed{str(int(reduce_speed_by*100))}_{exp_id}/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("\n" + "="*40)
print("         Experiment Configuration         ")
print("="*40)
print(f"City:                       {city}")
print(f"City Center Coordinates:    {city_center}")
print(f"Experiment ID:              {exp_id}")
print("-" * 40)
print(f"Radii (min → max, step):    {min_radius_km} km → {max_radius_km} km, step {radius_step_km} km")
print(f"Samples per Radius Circle:  {n_samples_circle}")
print(f"Threshold Distance [km]:    {th_distance_km}")
print("-" * 40)
print(f"Speed Reduction Factor:     {reduce_speed_by}")
print("-" * 40)
print(f"Penalization Factors (p):   {list_p}")
print(f"Epsilon Values (eps):       {list_eps}")
print(f"Number of Routes (k):       {k}")
print(f"k as Route Count:           {k_is_route_count}")
print("="*40)



if k_is_route_count:
    print(f"Using k as the number of desired routes (up to {k} routes will be computed in at most {max_it} iterations). [standard]")
else:
    print(f"Using k as the number of iterations (the penalization will run for exactly {k} iterations). [old version divercity]")
print("output folder", output_folder)
print("*************") 


# 1. Load City infos
#df_city = pd.read_csv("../data/city_info.csv")
#df_city_filtered = df_city[df_city["city"]==city][["lat", "lng"]].iloc[0]
#city_center = (df_city_filtered.lat, df_city_filtered.lng)


# 2. Load or Download the road network if not already present
network_file = f"../data/road_networks/{city}_{network_type}_{radius_city_m}.graphml.gz"
uncompressed_network_file = f"../data/road_networks/{city}_{network_type}_{radius_city_m}.graphml"

if os.path.exists(network_file):
    # If the compressed network file exists, decompress and load it
    print(f"Compressed network found: {network_file}")
    print(f"Decompressing and loading network...")
    print("-" * 40)
    with gzip.open(network_file, 'rb') as f_in:
        with open(uncompressed_network_file, 'wb') as f_out:
            f_out.write(f_in.read())
    
    # Load the graph from the decompressed file
    G = ox.load_graphml(uncompressed_network_file)

    # Optionally, remove the uncompressed file after loading
    os.remove(uncompressed_network_file)

else:
    
    print(f"Network file not found: {network_file}")
    print(f"Downloading road network for {city}...")
    print(f"Using center: {city_center} with radius {radius_city_m} meters")
    print("-" * 40)
    G = ox.graph_from_point(city_center, dist=radius_city_m, network_type=network_type, simplify=True)
    # Impute missing edge speeds and calculate edge travel times with the speed module
    G = ox.speed.add_edge_speeds(G)
    G = ox.speed.add_edge_travel_times(G)

    # Save the network as a regular GraphML file
    ox.save_graphml(G, uncompressed_network_file)

    # Compress the saved GraphML file
    with open(uncompressed_network_file, 'rb') as f_in:
        with gzip.open(network_file, 'wb') as f_out:
            f_out.writelines(f_in)

    # Optionally, remove the uncompressed file after compressing
    os.remove(uncompressed_network_file)


# save a thumbnail of the map
fig, ax = ox.plot_graph(G, node_size=0, show=False, close=True)
fig.savefig(output_folder+f"thumbnail_map_{city}.png", bbox_inches='tight')


# Reduce the speed on attractors (if specified)
if reduce_speed_by != 1:
    print("\n" + "="*40)
    print("         Speed Reduction on Attractors         ")
    print("="*40)
    print(f"Reducing speed on road types: ['motorway', 'trunk']")
    print(f"Speed reduction factor: {reduce_speed_by}")
    print("-" * 40)
    print(f"Identifying attractor edges...")
    
    attractor_edges = get_attractors_by_road_types(G, ["motorway", "trunk"])
    print(f"Number of edges affected: {len(attractor_edges)}")
    print(f"Applying speed reduction on attractors...")
    print("-" * 40)
    # Create a subgraph with only the attractor edges
    G_attractors = G.edge_subgraph(attractor_edges)
    fig, ax = ox.plot_graph(G_attractors, edge_color="red", edge_linewidth=1, node_size=0, show=False, close=False)
    fig.savefig(output_folder+f"attractors_{city}.png", bbox_inches='tight')
    
    for u, v, data in G.edges(data=True):
        data['backup_travel_time'] = data.get('travel_time')

    for u, v, key in attractor_edges:

        old_tt = G[u][v][key]["travel_time"]
        if reduce_speed_by>0:
            new_tt = old_tt/reduce_speed_by
        else:
            new_tt = np.inf
        G[u][v][key]["travel_time"] = new_tt

else:
    print("\n" + "="*40)
    print("         No Speed Reduction Applied         ")
    print("="*40)

# Convert the nx network to ig for faster operations
G_ig = ig.Graph.from_networkx(G)

# dict for quick ID lookup
node_nx_to_ig = {}
node_ig_to_nx = {}

for ind_n, n in enumerate(G_ig.vs()):
    node_nx_to_ig[n["_nx_name"]] = ind_n
    node_ig_to_nx[ind_n] = n["_nx_name"]

G_ig["info"] = {}
G_ig["info"]["node_nx_to_ig"] = node_nx_to_ig
G_ig["info"]["node_ig_to_nx"] = node_ig_to_nx

# Precompute edge lengths from the graph
edge_lengths = {(u, v): data["length"] for u, v, data in G.edges(data=True)}

print("\n" + "="*40)
print("         Road Network Information         ")
print("="*40)
print(f"Number of Nodes:           {len(G_ig.vs())}")
print(f"Number of Edges:           {len(G_ig.es())}")
print("-" * 40)

# 3. Perform the radial sampling    
r_list = np.arange(min_radius_km, max_radius_km+.01, radius_step_km)
sampling_info = perform_sampling(G, r_list, city_center, n_samples_circle, th_distance_km)

print("\n" + "="*40)
print("         Radial Sampling Information         ")
print("="*40)
print(f"Performing Radial Sampling with the following settings:")
print(f"Radii Range (km):          {min_radius_km} → {max_radius_km} (Step: {radius_step_km})")
print(f"Threshold Distance (km):   {th_distance_km}")
print(f"Samples per Circle:        {n_samples_circle}")
print("-" * 40)

print(f"Computed Radii List:       {r_list}")
print("="*40)


# plot the radial sampling
fig, ax = plt.subplots(figsize=(8, 8))
for r in sampling_info["gpd_points"]:
    sampling_info["gpd_points"][r].plot(ax=ax, color="lightblue", markersize=5)
count_outside = 0
for r in sampling_info["points_outside"]:
    for po in sampling_info["points_outside"][r]:
        plt.scatter(po[0], po[1], marker="x", color="red", s=50)
        count_outside+=1
plt.scatter([], [], marker='x', color='red', label=f'# {count_outside}')
plt.legend()
plt.savefig(output_folder+"fig_sampling.pdf", bbox_inches='tight')



# 4. Compute the alternative routes using path penalization
routing_elements = sampling_info["sampled_nodes"]

manager = multiprocessing.Manager()
penalized_paths_dict = manager.dict()
processes = []

for r in routing_elements.keys():

    process = multiprocessing.Process(target=compute_path_penalization_r, 
                                      args=(G_ig, r, list_p, k, routing_elements, penalized_paths_dict, attribute, k_is_route_count, max_it))

    
    
    process.start()
    processes.append(process)
    
for p in processes:
    p.join()

for p in processes:
    p.close()

    
penalized_paths_dict = dict(penalized_paths_dict)

# 4b. Save the generated paths (in case of some errors the slow part is stored)
if save_routes == 1:
    print("\n" + "="*40)
    print("         Route Computation Completed         ")
    print("="*40)
    print(f"Saving generated routes to file: {output_folder}generated_routes.lz4")
    print("-" * 40)
    with lz4.frame.open(output_folder+f"generated_routes.lz4", "wb") as f:
        packed_data = msgpack.packb(penalized_paths_dict)
        f.write(packed_data)


# 4c. Save the sampling infos, jaccard and the generated paths
_ = sampling_info.pop("gpd_points")

output_file = open(output_folder+"sampling_info.json", "w")
json.dump(sampling_info, output_file)
output_file.close()


# 5. Compute measures on the computed paths
manager = multiprocessing.Manager()
results_dict = manager.dict()
processes = []

for r in routing_elements.keys():

    process = multiprocessing.Process(target=parallel_compute_divercity_score_weighted,
                                      args=(routing_elements, r, k, penalized_paths_dict, n_samples_circle, list_p, list_eps, edge_lengths, results_dict))
    
    process.start()
    processes.append(process)
    
for p in processes:
    p.join()

for p in processes:
    p.close()


print("\n" + "="*40)
print("         Results Computation Completed         ")
print("="*40)
print(f"Saving results to file: {output_folder}results.json")
print("-" * 40)

results_dict = dict(results_dict)

# 6 save the dictionary that contains the computed results (jaccard)
output_file = open(output_folder+"results.json", "w")
json.dump(results_dict, output_file)
output_file.close()