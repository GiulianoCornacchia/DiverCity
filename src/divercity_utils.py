import geopandas as gpd
from geopy import distance
from shapely.geometry import Point
import geopandas as gpd
from scipy.spatial import cKDTree
import numpy as np

import osmnx as ox

import warnings
warnings.filterwarnings("ignore")

from timeit import default_timer as timer
import subprocess



def compute_divercity_city(city_name, city_center, exp_id, list_p, list_eps, k, attribute, 
                                 min_radius_km, max_radius_km, radius_step_km, 
                                 n_samples_circle, save_routes):

    """
    Compute DiverCity metrics for a single city.

    This function computes route diversification and evaluates DiverCity metrics for the 
    specified city using the given parameters. It builds a command line argument string 
    for the `compute_divercity_osm.py` script and executes it using `subprocess.Popen`.

    Args:
        city_name (str): Name of the city for which DiverCity metrics are computed.
        city_center (tuple): Tuple containing latitude and longitude of the city center.
        exp_id (str): Unique experiment identifier for result storage and tracking.
        list_p (list of float): List of penalization factors for Path Penalization (PP).
        list_eps (list of float): List of epsilon values for identifying Near-Shortest Routes (NSR).
        k (int): Number of alternative routes to compute.
        attribute (str): Path attribute to optimize (e.g., "traveltime").
        min_radius_km (int): Starting radius (in km) for radial sampling.
        max_radius_km (int): Ending radius (in km) for radial sampling.
        radius_step_km (int): Step size (in km) for concentric circles in radial sampling.
        n_samples_circle (int): Number of samples per circle for radial sampling.
        save_routes (int): Flag to save generated routes (1 = Yes, 0 = No).
    
    Returns:
        None. The function prints the execution time of the computation.

    Raises:
        subprocess.CalledProcessError: If the command execution fails.
        ValueError: If the input parameters are invalid.
    """
    
    
    # Record the start time
    start_time = timer()

    str_list_p = str(list_p).replace(" ","")
    str_list_eps = str(list_eps).replace(" ","")
    
    opts = f"-c {city_name} --lat {city_center[0]} --lng {city_center[1]} -i {exp_id} --plist {str_list_p} -k {k} --epslist {str_list_eps} -a {attribute} --rfrom {min_radius_km} --rto {max_radius_km} --rstep {radius_step_km} -n {n_samples_circle} --saveroutes {save_routes} --njobs 10"
    
    command_list = ['python', "compute_divercity_osm.py"] + opts.split(" ")
    
    process = subprocess.Popen(command_list)
    process.wait()
    
    # Record the end time
    end_time = timer()
    execution_time = end_time - start_time



EARTH_RADIUS = 6371000  # Radius of the Earth in meters

def convert_to_cartesian(lng, lat):
    """
    Convert geographic coordinates (longitude, latitude) to Cartesian coordinates (x, y)
    using an equirectangular projection.

    Parameters:
    -----------
    lng : float
        The longitude in degrees.
    lat : float
        The latitude in degrees.

    Returns:
    --------
    tuple
        A tuple (x, y) representing the Cartesian coordinates in meters, where `x` is the 
        horizontal distance and `y` is the vertical distance from the equator.

    Example:
    --------
    >>> convert_to_cartesian(-73.985428, 40.748817)
    (-8240216.806799497, 4480358.041798708)

    Notes:
    ------
    - This conversion assumes the Earth is a perfect sphere and uses an equirectangular 
      projection, which works best for small distances or areas.
    - The result represents meters relative to the Earth's radius.
    """
    
    lat_rad = np.radians(lat)
    lng_rad = np.radians(lng)
    
    x = EARTH_RADIUS * np.cos(lat_rad) * lng_rad
    y = EARTH_RADIUS * lat_rad
    
    return x, y



def perform_sampling(G, r_list, city_center, n_samples_circle, th_distance):#, kd_tree):
    
    
    # GeoPandas representing the sampled points
    gpd_points = {}

    # (lng, lat) representing the sampled points
    sampled_points = {}

    # closest graph node associated with the sampled points
    sampled_nodes = {}

    # sampled nodes coordinates
    sampled_nodes_coordinates = {}

    # list of points that don't have a graph node within th_distance km
    points_outside = {}

    # distance point to closest node
    sampled_node_distance = {}

    # Create a KD-tree
    list_node_ids_kdt = list(G.nodes())
    node_coordinates = [(data['x'], data['y']) for node, data in G.nodes(data=True)]
    # Convert all node coordinates to the Cartesian projection
    cartesian_coordinates = [convert_to_cartesian(lon, lat) for lon, lat in node_coordinates]
    # Create the KDTree using the Cartesian coordinates
    kd_tree = cKDTree(cartesian_coordinates)
        

    for r in r_list:

        points_r_km = fixed_radius_sampling(city_center, r, n_samples_circle)
        sampled_points[r] = points_r_km

        edges_r_km, nodes_r_km, points_outside_r, sampled_node_distance_r = [], [], [], []

        for (lng, lat) in points_r_km:

            # retrieve closest Graph node
            query_point_cartesian = convert_to_cartesian(lng, lat)
            dist, nearest_node_index = kd_tree.query(query_point_cartesian)
            node_id_nearest_node = list_node_ids_kdt[nearest_node_index]

            node_data = G.nodes[node_id_nearest_node]
            lng_node = node_data["x"]
            lat_node = node_data["y"]
            
            haversine_distance_km = ox.distance.great_circle(lat, lng, lat_node, lng_node)/1e3

            if haversine_distance_km <= th_distance:
                nodes_r_km.append(node_id_nearest_node)

            else:
                points_outside_r.append([lng, lat])
                nodes_r_km.append({})

            sampled_node_distance_r.append(haversine_distance_km)
            sampled_nodes_coordinates[node_id_nearest_node] = (lng_node, lat_node)
            

        sampled_nodes[r] = nodes_r_km
        points_outside[r] = points_outside_r
        sampled_node_distance[r] = sampled_node_distance_r
        

        geometry = [Point(xy) for xy in points_r_km]
        gdf_points = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")

        gpd_points[r] = gdf_points
        
        
        sampling_info = {}

        sampling_info["sampling_parameters"] = {"r_list": list(r_list), 
                                                "n_samples_circle": n_samples_circle, 
                                                "th_distance": th_distance}

        sampling_info["sampled_points"] = sampled_points
        sampling_info["gpd_points"] = gpd_points
        
        sampling_info["sampled_nodes"] = sampled_nodes
        sampling_info["points_outside"] = points_outside
        sampling_info["sampled_node_distance"] = sampled_node_distance
        sampling_info["sampled_nodes_coordinates"] = sampled_nodes_coordinates

        
    return sampling_info




def fixed_radius_sampling(center, radius, nb_samples):
    """
    Generate a fixed number of equidistant sample points on the circumference of a circle 
    defined by a given center and radius.

    Parameters:
    -----------
    center : tuple
        A tuple containing the coordinates (latitude, longitude) of the center point.
    radius : float
        The radius of the circle in kilometers on which the sample points will be placed.
    nb_samples : int
        The number of sample points to generate on the circle's circumference.

    Returns:
    --------
    list of tuple
        A list of tuples where each tuple represents the coordinates (longitude, latitude) 
        of a sample point on the circle's circumference.
    
    Example:
    --------
    >>> fixed_radius_sampling((40.748817, -73.985428), 5, 12)
    [(-73.985428, 40.803654), (-73.930601, 40.800425), ...]

    Notes:
    ------
    - The function uses geographic coordinates and assumes the Earth is a perfect sphere, 
      so it provides approximate results over long distances.
    - The output list contains `nb_samples` points distributed evenly in a circular pattern 
      around the center point.
    """
    
    res = []

    for theta in range(nb_samples):
        point = distance.distance(kilometers=radius).destination(center, theta * (360/nb_samples))
        res = res + [(point[1], point[0])]

    return res




def filter_near_shortest(path_list, eps):
    
    max_cost = path_list[0]["original_cost"] * (1+eps)
    near_shortest_paths = [p["node_list_nx"] for p in path_list if p["original_cost"] <= max_cost]
    
    return near_shortest_paths

def get_pair_list(route):
    consecutive_pairs = [(route[i], route[i+1]) for i in range(len(route)-1)]
    return consecutive_pairs


def jaccard_pairwise_weighted(path_list, dict_weight):
     
    list_jaccard = []
    
    if len(path_list) <= 1:
        return 1

    for ind_i, path_i in enumerate(path_list):
        route_i = get_pair_list(path_i)
        for ind_j, path_j in enumerate(path_list):
            route_j = get_pair_list(path_j)
            if ind_i<ind_j:
                jacc = weighted_jaccard_similarity(route_i, route_j, dict_weight)         
                list_jaccard.append(jacc)
                
    return list_jaccard


def jaccard_pairwise_weighted_dict(path_list, dict_weight):
     
    dict_jaccard = {}
    
    if len(path_list) <= 1:
        return 1

    for ind_i, path_i in enumerate(path_list):

        route_i = get_pair_list(path_i)
        for ind_j, path_j in enumerate(path_list):
            route_j = get_pair_list(path_j)

            if ind_i<ind_j:
                jacc = weighted_jaccard_similarity(route_i, route_j, dict_weight)         
                dict_jaccard[ind_i, ind_j] = jacc
                
    return dict_jaccard


def weighted_jaccard_similarity(list1, list2, dict_weights):
    
    set1 = set(list1)
    set2 = set(list2)
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    # Calculate total length of edges in intersection and union
    intersection_length = sum(dict_weights[edge] for edge in intersection)
    union_length = sum(dict_weights[edge] for edge in union)
    
    weighted_jaccard = intersection_length / union_length if union_length > 0 else 0
    
    return weighted_jaccard






def parallel_compute_divercity_score_weighted(routing_elements, r, max_k, dict_paths_pp, n_points_circle, list_p, list_eps, dict_weight, results_dict):
    

    r = float(r)
    
    dict_scores = {}

    n_points_circle = len(routing_elements[r])

    for ind_src in range(n_points_circle):

        for ind_dest in range(n_points_circle):

            key_od = f"{ind_src}_{ind_dest}"

            if key_od in dict_paths_pp[r]:

                dict_scores[key_od] = {}

                for eps in list_eps:

                    dict_scores[key_od][eps] = {}

                    for p in list_p:
                        
                        try:
                            list_paths = dict_paths_pp[r][key_od][p]
                        except:
                            list_paths = dict_paths_pp[r][key_od][str(p)]

                        # len all paths
                        number_alternative_routes = len(list_paths)
                        
                        # len NSP
                        near_shortest_paths = filter_near_shortest(list_paths, eps)
                        number_NSP = len(near_shortest_paths)
                        
                        # jaccard pw NSP
                        jaccard_pw = jaccard_pairwise_weighted(near_shortest_paths, dict_weight)
                        avg_jaccard_pw = np.mean(jaccard_pw)
                        
                        divercity = (1-avg_jaccard_pw) * number_NSP

                        # list iterations
                        list_iterations = [p["iteration"] for p in list_paths]
                        

                        dict_scores[key_od][eps][p] = {"divercity":divercity,
                                                        "avg_jaccard": avg_jaccard_pw, 
                                                        "number_paths":number_alternative_routes,
                                                        "number_nsp": number_NSP,
                                                        "list_iterations": list_iterations}


                        # Analysis "at k". It means compute the measures on the first k paths

                        # jaccard Matrix to speed up the jaccard
                        all_paths_list = [p["node_list_nx"] for p in list_paths]
                        matrix_jaccard = jaccard_pairwise_weighted_dict(all_paths_list, dict_weight)

                        list_at_k = list(np.arange(2, max_k+1, 1))
                        list_nsp_at_k = []
                        list_jaccard_at_k = []
                        
                        for at_k in np.arange(2, max_k+1, 1):
                        
                            list_paths_at_k = list_paths[:at_k]

                            # NSP at k
                            near_shortest_paths_at_k = filter_near_shortest(list_paths_at_k, eps)
                            number_NSP_at_k = len(near_shortest_paths_at_k)
                            list_nsp_at_k.append(number_NSP_at_k)
                        
                            # get the jaccard from the matrix
                            list_indices = [all_paths_list.index(p) for p in near_shortest_paths_at_k]
                            list_jaccards_matrix = [matrix_jaccard[ind_i, ind_j] for ind_i in list_indices for ind_j in list_indices if ind_i < ind_j]
                            avg_jaccard_pw_at_k = np.mean(list_jaccards_matrix)
                            list_jaccard_at_k.append(avg_jaccard_pw_at_k)
                        

                        # add at_k
                        for at_k, nsp_at_k, jaccard_at_k in zip(np.arange(2, max_k+1, 1), list_nsp_at_k, list_jaccard_at_k):
                            dict_scores[key_od][eps][p][f"number_nsp_at_k_{at_k}"] = nsp_at_k
                            dict_scores[key_od][eps][p][f"avg_jaccard_at_k_{at_k}"] = jaccard_at_k

    results_dict[r] = dict_scores

    return


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