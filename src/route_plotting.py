import matplotlib.pyplot as plt
import geopandas as gpd
from collections import Counter
from shapely.geometry import box
from divercity_utils import get_pair_list
from shapely.ops import linemerge
from collections import Counter
from geopy.distance import distance
from shapely.geometry import Polygon, Point
import pandas as pd
import os
import osmnx as ox
from results_utils import process_city, filter_by_p_eps
import gzip
from divercity_utils import get_attractors_by_road_types
import json


def visualize_alternative_routes(alternatives, gpd_edges, dict_edges_geo, eps=0.3, max_k=10, ax=None, plot_road_network=True, filter_road_types=[]):
    """
    Visualizes the NSR (Non-Shortest Route) and non-NSR routes from given alternative routes.

    Parameters:
    - alternatives: List of route dictionaries containing 'node_list_nx' and 'original_cost'.
    - gpd_edges: GeoDataFrame of the road network edges.
    - dict_edges_geo: Dictionary mapping node pairs to their geographical edges.
    - eps: Threshold factor for determining non-NSR routes (default 0.3, meaning 1.3*original cost).
    - max_k: Number of alternative routes to consider (default 10).
    - ax: Matplotlib axis to draw on. If None, a new figure is created.
    - plot_road_network: Boolean flag to plot the underlying road network (default True).
    - filter_road_types: List of road types to filter out from the background network (default []).
    """
    
    gdf_sp = from_route_to_gdf_dict(alternatives[0]["node_list_nx"], dict_edges_geo)
    max_cost = alternatives[0]["original_cost"] * (1 + eps)
    
    list_visits_NSR = []
    list_visits_nonNSR = []
    
    for p in alternatives[:max_k]:
        gdf_route = from_route_to_gdf_dict(p["node_list_nx"], dict_edges_geo)  
        
        route = p["node_list_nx"]
        edge_list = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
        
        if p["original_cost"] >= max_cost:
            list_visits_nonNSR += edge_list
        else:
            list_visits_NSR += edge_list
    
    value_counts_NSR = dict(Counter(list_visits_NSR))
    value_counts_nonNSR = dict(Counter(list_visits_nonNSR))
    
    gdf_routes = gpd_edges.copy(deep=True)
    
    gdf_routes["measure_NSR"] = [value_counts_NSR.get((u, v), 0) for u, v in zip(gdf_routes["u"], gdf_routes["v"])]
    gdf_routes["measure_nonNSR"] = [value_counts_nonNSR.get((u, v), 0) for u, v in zip(gpd_edges["u"], gpd_edges["v"])]
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    
    gdf_NSR = gdf_routes[gdf_routes["measure_NSR"] > 0]
    gdf_NSR.plot(color="darkblue", linewidth=gdf_NSR["measure_NSR"], alpha=1, zorder=3, ax=ax)
    
    gdf_nonNSR = gdf_routes[gdf_routes["measure_nonNSR"] > 0]
    if len(gdf_nonNSR) > 0:
        gdf_nonNSR.plot(color="red", linewidth=gdf_nonNSR["measure_nonNSR"], alpha=1, zorder=3, ax=ax)
    
    xlim, ylim = ax.set_xlim(), ax.set_ylim()
    
    if plot_road_network:
        bbox = box(xlim[0], ylim[0], xlim[1], ylim[1])
        clipped_gdf = gpd.clip(gpd_edges, bbox)
        
        clipped_gdf["to_discard"] = clipped_gdf["highway"].apply(
            lambda x: bool(set(x).intersection(filter_road_types)) if isinstance(x, list) else x in filter_road_types
        )
        
        clipped_gdf = clipped_gdf[clipped_gdf["to_discard"] == False]
        clipped_gdf.plot(ax=ax, color="#BFBFBF", linestyle="-", linewidth=.1, alpha=1, zorder=1)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis("off")
    
    return ax



def from_route_to_gdf_dict(route, dict_edges):
    
    edge_list = get_pair_list(route)
    list_line_string = []
    
    for (u, v) in edge_list:
        line = dict_edges[u, v]
        list_line_string.append(line)
    
    # Merge the LineString objects into a single LineString
    merged_line = linemerge(list_line_string)
    # Create a GeoDataFrame with the merged LineString
    gdf = gpd.GeoDataFrame(geometry=[merged_line])
    gdf = gdf.set_crs(epsg=4326)

    return gdf




def compute_gpd_divercity_nodes(df_city, dict_sampling, p=0.1, eps=0.3, k=10):

    list_all_geometries = []
    list_all_values = []
    
    df_p_eps = filter_by_p_eps(df_city, p, eps)

    nodes_in_r = list(set(list(df_p_eps["origin_idx"].unique()) + list(df_p_eps["dest_idx"].unique())))
    nodes_in_r = list(set(list(df_p_eps["origin_node"].unique()) + list(df_p_eps["dest_node"].unique())))

    for node in nodes_in_r:

        node_divercity = df_p_eps[(df_p_eps["origin_node"]==node)|(df_p_eps["dest_node"]==node)][f"div_k_{k}"].mean()
        list_all_geometries.append(Point(dict_sampling["sampled_nodes_coordinates"][str(node)]))
        list_all_values.append(node_divercity)
            
    gpd_nodes = gpd.GeoDataFrame(geometry=list_all_geometries, crs="EPSG:4326")
    gpd_nodes["divercity"] = list_all_values

    return gpd_nodes



# Function to process a single city and return its results
def process_single_city(city, radius_m, exp_id, k):
    # Initialize results dictionary
    city_results = {}

    # Load the DiverCity
    df_cities = pd.read_csv("../data/city_info.csv")
    directory_path = '../data/plots/'
    output_folder = directory_path + city + "/"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    df_city_filtered = df_cities[df_cities["city"] == city][["lat", "lng"]].iloc[0]
    city_center = (df_city_filtered.lat, df_city_filtered.lng)
    
    # Prepare the DiverCity results
    _, df_city = process_city(city, f"../data/results/{city}_{exp_id}/", max_k=k)
    city_results['divercity'] = df_city
    
    # Load the road network
    network_file = f"../data/road_networks/{city}_drive_{radius_m}.graphml.gz"
    uncompressed_network_file = f"../data/road_networks/{city}_drive_{radius_m}.graphml"
    
    with gzip.open(network_file, 'rb') as f_in:
        with open(uncompressed_network_file, 'wb') as f_out:
            f_out.write(f_in.read())
    
    # Load the graph from the decompressed file
    G = ox.load_graphml(uncompressed_network_file)
    
    # Extract the attractors
    attractors = get_attractors_by_road_types(G, ["motorway", "trunk"])
    G_attractors = G.edge_subgraph(attractors)
    
    attractor_edges = ox.graph_to_gdfs(G_attractors, nodes=False)
    attractor_edges = attractor_edges.reset_index()
    
    # Load sampling info
    with open(f'../data/results/{city}_{exp_id}/sampling_info.json', 'r') as file:
        dict_sampling = json.load(file)
    
    city_results['road_networks'] = {
        "attractor_edges": attractor_edges,
        "dict_sampling": dict_sampling
    }
    
    del G
    return city, city_results


def create_geodesic_circle(center, radius_km, nb_samples=360):
    """
    Creates a geodesic circle around a central point.
    
    Parameters:
    - center: tuple (latitude, longitude) of the central point.
    - radius_km: float, the radius of the circle in kilometers.
    - nb_samples: int, number of points used to approximate the circle (default: 360).
    
    Returns:
    - A Shapely Polygon representing the circle.
    """
    # Generate points around the circle
    points = [
        distance(kilometers=radius_km).destination(center, theta * (360 / nb_samples))
        for theta in range(nb_samples)
    ]
    # Create a polygon from the points
    circle_polygon = Polygon([(point[1], point[0]) for point in points])  # (lon, lat)
    return circle_polygon