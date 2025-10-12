import math
import json
import os
import numpy as np
import osmnx as ox
import networkx as nx
import gzip
from divercity_utils import get_attractors_by_road_types, convert_to_cartesian
from scipy.spatial import cKDTree
import geopandas as gpd
from geopy.distance import distance
from shapely.geometry import Polygon, Point
import matplotlib.colors as mcolors
import json
from matplotlib import pyplot as plt


# Function to calculate the bearing between two points
def calculate_bearing(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    initial_bearing = math.atan2(x, y)
    # Convert bearing from radians to degrees and normalize it
    bearing = (math.degrees(initial_bearing) + 360) % 360
    return bearing
    

def compute_edge_capacity(speed_km_h, num_lanes):

    conversion_factor = 2.2369362912
    q = 0.5

    speed_m_s = speed_km_h/3.6
    sl = speed_m_s*conversion_factor

    # when the speed limit of a road segment sl≤45, it is defined as an arterial road
    if sl <= 45:
        capacity = 1900*num_lanes*q
    # when the speed limit of a road segment 45<sl<60, it is defined as a highway
    elif 45<sl<60:
        capacity = (1000+20*sl)*num_lanes
    # when the speed limit of a road segment sl ≥60, it is defined as a freeway
    elif sl>=60:
        capacity = (1700+10*sl)*num_lanes

    return capacity
        


def get_distance_edges_center(G, city_center, node_to_coords, as_dict=False):
    
    list_dist_center = []
    dict_dist_center = {}

    for edge in G.edges(data=True):
    
        lng_u, lat_u = node_to_coords[edge[0]]
        lng_v, lat_v = node_to_coords[edge[1]]
    
        distance_u_km = ox.distance.great_circle(city_center[0], city_center[1], lat_u, lng_u)/1e3
        distance_v_km = ox.distance.great_circle(city_center[0], city_center[1], lat_v, lng_v)/1e3
        mean_dist = np.mean([distance_u_km, distance_v_km])

        list_dist_center.append(mean_dist)

        if as_dict:
            dict_dist_center[edge[0], edge[1]] = mean_dist

    if as_dict:
        return dict_dist_center
        
    return list_dist_center


def get_attractors_by_road_types(G, attractor_types):

    # Extract edges that meet the criteria
    attractor_edges = []
    for u, v, key, data in G.edges(keys=True, data=True):

        highway_type = data.get("highway", None)
        if isinstance(highway_type, list):
            set_highway_type = set(highway_type)
        else:
            set_highway_type = set([highway_type])

        if set(attractor_types) & set(set_highway_type):
            attractor_edges.append((u, v, key))


    return attractor_edges


# Define a function to calculate the midpoint of an edge
def calculate_midpoint(geometry):
    if isinstance(geometry, LineString):
        # If geometry is a LineString (common for OSMnx edges), calculate the midpoint
        midpoint = geometry.interpolate(0.5, normalized=True)  # Midpoint at 50%
        return midpoint
    else:
        # If not a LineString, return the first point (fallback)
        return geometry.centroid



def compute_network_measures(city, city_center, network_type, radius_city_m, save=False, saveFig=False, filename=""):

    # Load the existing road network
    network_file = f"../data/road_networks/{city}_{network_type}_{radius_city_m}.graphml.gz"
    uncompressed_network_file = f"../data/road_networks/{city}_{network_type}_{radius_city_m}.graphml"
    
    if os.path.exists(network_file):
        # If the compressed network file exists, decompress and load it
        print(f"Loading compressed network from {network_file}...")
        with gzip.open(network_file, 'rb') as f_in:
            with open(uncompressed_network_file, 'wb') as f_out:
                f_out.write(f_in.read())
        
        # Load the graph from the decompressed file
        G = ox.load_graphml(uncompressed_network_file)
    
        # Optionally, remove the uncompressed file after loading
        os.remove(uncompressed_network_file)
    
    else:
        print(f"Network {network_file} does not exist. Downloading it using the notebook '1_Download_Road_Network.ipynb'")
        return
       

    # add the bearing
    G = ox.bearing.add_edge_bearings(G)

    # dict node to coordinates
    node_to_coords = {}
    for node in G.nodes(data=True):
        node_to_coords[node[0]] = [node[1]["x"], node[1]["y"]]
    
    # basic stats
    basic_stats = ox.basic_stats(G)

    # attractors
    attractor_edges = get_attractors_by_road_types(G, ["motorway", "trunk"])
    G_attractors = G.edge_subgraph(attractor_edges)

    if saveFig:
        fig, ax = ox.plot_graph(G, edge_color="lightgray", edge_linewidth=0.1, show=False, close=False, node_size=0, bgcolor="white")
        if len(G_attractors.edges())>0:
            ox.plot_graph(G_attractors, edge_color="red", edge_linewidth=1, ax=ax, node_size=0, show=False, close=False)
        plt.savefig(f"{filename}map_attractors_type_{city}.png", dpi=600,  bbox_inches='tight')
    
    
    # INFO of ALL the edges (length, capacity, max_speed_kph, distance_to_center_km, bearing)
    dict_edge_info = {}
    dict_distance_edges_to_center = get_distance_edges_center(G, city_center, node_to_coords, as_dict=True)
    
    for u, v, key, data in G.edges(keys=True, data=True):
        
        maxspeed = data.get("speed_kph", None)
        lanes = data.get("lanes", None)
        bearing =  data.get("bearing", None)
    
        if isinstance(lanes, list):
            lanes = lanes[0]
        if isinstance(lanes, str):
            try:
                lanes = int(lanes)
            except ValueError:
                lanes = 1
    
        if lanes is None:
            lanes = 1
            
    
        dict_edge_info[f"{u}_{v}"] = {"length": data["length"],
                                      "max_speed_kph" :maxspeed,
                                      "distance_to_center_km": dict_distance_edges_to_center[u,v],
                                     "bearing": bearing}
    
    
    dict_measures = {"note": "version for DiverCity october 2025.",
                     
                    "basic_stats":basic_stats,
    
                     "info_attractors": {"all_edges":list(G_attractors.edges())},
    
                     "info_edges": dict_edge_info}

                     
                
    if save:
        with open(f"{filename}info_{city}.json", 'w') as json_file:
            json.dump(dict_measures, json_file)


    return dict_measures



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


def calculate_point_to_edge_distances(G, points, return_edges=False):
    
    x_coords, y_coords = points.x.values, points.y.values
    closest_edges = ox.distance.nearest_edges(G, x_coords, y_coords)
    
    closest_distances = []
    for x, y, edge in zip(x_coords, y_coords, closest_edges):
        edge_geom = G.edges[edge].get('geometry', None)
        if edge_geom is None:
            edge_geom = Point(G.nodes[edge[0]]["x"], G.nodes[edge[0]]["y"]).buffer(0.001).boundary
            
        closest_distances.append(Point(x, y).distance(edge_geom) / 1000)

    if return_edges:
        closest_distances, closest_edges
        
    return closest_distances


# The attractor spatial dispersion (H)
def compute_attractor_spatial_dispersion(city, city_center, radius_m, list_n_samples, list_dist_kdtree, save=False, saveFig=False):

    output_folder = f"../data/attractor_spatial_dispersion/{city}/"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load the road network
    network_file = f"../data/road_networks/{city}_drive_{radius_m}.graphml.gz"
    uncompressed_network_file = f"../data/road_networks/{city}_drive_{radius_m}.graphml"
    
    with gzip.open(network_file, 'rb') as f_in:
        with open(uncompressed_network_file, 'wb') as f_out:
            f_out.write(f_in.read())
    
    # Load the graph from the decompressed file
    G = ox.load_graphml(uncompressed_network_file)

    # Optionally, remove the uncompressed file after loading
    os.remove(uncompressed_network_file)

    # extract the attractors
    attractors = get_attractors_by_road_types(G, ["motorway", "trunk"])
    G_attractors = G.edge_subgraph(attractors)
    attractor_edges = ox.graph_to_gdfs(G_attractors, nodes=False)
    attractor_edges = attractor_edges.reset_index()
    
    # Project the graph to UTM
    G_attractors_projected = ox.project_graph(G_attractors)
    utm_crs = G_attractors_projected.graph['crs']  # Extract the CRS of the projected graph
    
    # Create the KDTree using the Cartesian coordinates
    list_node_ids_kdt = list(G.nodes())
    node_coordinates = [(data['x'], data['y']) for node, data in G.nodes(data=True)]
    cartesian_coordinates = [convert_to_cartesian(lon, lat) for lon, lat in node_coordinates]
    kd_tree = cKDTree(cartesian_coordinates)
    
    # Create a ring ( km of radius) as bound
    gpd_ring = gpd.GeoDataFrame(geometry=[create_geodesic_circle(city_center, radius_m/1000, nb_samples=360)],  crs="EPSG:4326")
    W, S, E, N = gpd_ring.total_bounds
    
    dict_distances = {}
    
    for n_samples in list_n_samples:
                      
        dict_distances[n_samples] = {}
    
        for radius_kdtree in list_dist_kdtree:
    
            dict_distances[n_samples][radius_kdtree] = {}
    
            selected_points = []
            discarded_points = []
            
            while len(selected_points)<n_samples:
                
                random_lats = np.random.uniform(S, N, n_samples)
                random_lons = np.random.uniform(W, E, n_samples)
            
                for (r_lat, r_lng) in zip(random_lats, random_lons):
                    if gpd_ring.contains(Point(r_lng, r_lat)).values[0]:
                        query_point_cartesian = convert_to_cartesian(r_lng, r_lat)
                        if len(kd_tree.query_ball_point(query_point_cartesian, radius_kdtree, return_sorted=True))>0:
                             selected_points.append([r_lng, r_lat])
                        else:
                            discarded_points.append([r_lng, r_lat])
    
            random_points = gpd.GeoSeries([Point(lon, lat) for (lon, lat) in selected_points], crs="EPSG:4326")
            discarded_points = gpd.GeoSeries([Point(lon, lat) for (lon, lat) in discarded_points], crs="EPSG:4326")
            
    
            random_points_projected = random_points.to_crs(utm_crs)
            G_net_projected = ox.project_graph(G)
    
            distances_to_attr = calculate_point_to_edge_distances(G_attractors_projected, random_points_projected)
            dict_distances[n_samples][radius_kdtree]["distances"] = distances_to_attr
    
            dict_distances[n_samples][radius_kdtree]["n_valid_points"] = len(random_points)
            dict_distances[n_samples][radius_kdtree]["n_invalid_points"] = len(discarded_points)

            if saveFig:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                attractor_edges.plot(color="black", ax=ax)
                random_points.plot(color="blue", ax=ax, markersize=.5, label="valid")
                discarded_points.plot(color="red", ax=ax, markersize=.5, label="discarded")
                plt.legend()
                fig.savefig(f"{output_folder}map_{city}_samples{n_samples}_{radius_kdtree}.png", bbox_inches='tight')

            if saveFig:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
                gpd_color_points = gpd.GeoDataFrame(geometry=[Point(lon, lat) for (lon, lat) in selected_points])
                gpd_color_points["distance"] = distances_to_attr
                
                gpd_color_points.plot(markersize=1, column="distance", cmap="coolwarm_r", ax=ax)
                norm = mcolors.Normalize(vmin=gpd_color_points["distance"].min(), vmax=gpd_color_points["distance"].max())          
                cbar = plt.cm.ScalarMappable(norm=norm, cmap="coolwarm_r")
                cbar.set_array([])  # Required for colorbar creation
                # Add the colorbar to the right side
                cbar_ax = fig.colorbar(cbar, ax=ax, orientation="vertical", fraction=0.03, pad=0.1)
                cbar_ax.set_label("Distance to Attractor (km)", fontsize=10)
                attractor_edges.plot(color="black", ax=ax)
                fig.savefig(f"{output_folder}map_{city}_distance{n_samples}_{radius_kdtree}.png", bbox_inches='tight')

    if save:
        output_file = open(f"{output_folder}attractor_spatial_dispersion_{city}.json", "w")
        json.dump(dict_distances, output_file)
        output_file.close()
    
    return dict_distances




def compute_distance_node_to_attractors_city(city, center_point, radius_m, exp_id):

    output_folder = f"../data/distance_node_to_attractors/{city}/"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load the road network
    network_file = f"../data/road_networks/{city}_drive_{radius_m}.graphml.gz"
    uncompressed_network_file = f"../data/road_networks/{city}_drive_{radius_m}.graphml"
    
    with gzip.open(network_file, 'rb') as f_in:
        with open(uncompressed_network_file, 'wb') as f_out:
            f_out.write(f_in.read())
    
    # Load the graph from the decompressed file
    G = ox.load_graphml(uncompressed_network_file)

    # Optionally, remove the uncompressed file after loading
    os.remove(uncompressed_network_file)
    
    # extract the attractors
    attractors = get_attractors_by_road_types(G, ["motorway", "trunk"])
    G_attractors = G.edge_subgraph(attractors)
    attractor_edges = ox.graph_to_gdfs(G_attractors, nodes=False)
    attractor_edges = attractor_edges.reset_index()
    
    # Project the graph to UTM
    G_attractors_projected = ox.project_graph(G_attractors)
    utm_crs = G_attractors_projected.graph['crs']  # Extract the CRS of the projected graph
    
    with open(f'../data/results/{city}_{exp_id}/sampling_info.json', 'r') as file:
        dict_sampling = json.load(file)
    
    list_r = []
    list_id = []
    list_geometry = []
    
    for r in dict_sampling["sampled_nodes"]:
        for node in dict_sampling["sampled_nodes"][r]:
            if node != {}:
                node_data = G.nodes[node]
                node_point = (node_data["x"], node_data["y"])
    
                list_r.append(r)
                list_id.append(node)
                list_geometry.append(node_point)
    
    
    node_points = gpd.GeoSeries([Point(lon, lat) for (lon, lat) in list_geometry], crs="EPSG:4326")
    node_points_projected = node_points.to_crs(utm_crs)
    
    distances_to_attr, closest_edges = calculate_point_to_edge_distances(G_attractors_projected, node_points_projected, return_edges=True)
    dict_node_dist = {node_id: dist for node_id, dist in zip(list_id, distances_to_attr)}
    
    output_file = open(f"{output_folder}distance_node_to_attractors_{city}.json", "w")
    json.dump(dict_node_dist, output_file)
    output_file.close()




def calculate_point_to_edge_distances(G, points, return_edges=False):
    
    x_coords, y_coords = points.x.values, points.y.values
    closest_edges = ox.distance.nearest_edges(G, x_coords, y_coords)
    
    closest_distances = []
    for x, y, edge in zip(x_coords, y_coords, closest_edges):
        edge_geom = G.edges[edge].get('geometry', None)
        if edge_geom is None:
            edge_geom = Point(G.nodes[edge[0]]["x"], G.nodes[edge[0]]["y"]).buffer(0.001).boundary
            
        closest_distances.append(Point(x, y).distance(edge_geom) / 1000)

    if return_edges:
        return closest_distances, closest_edges
        
    return closest_distances

