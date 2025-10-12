import matplotlib.pyplot as plt
from collections import Counter
from divercity_utils import filter_near_shortest
from shapely.geometry import Point, LineString
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import networkx as nx
import igraph as ig
from routing_utils import compute_path_penalization_r

import multiprocessing
import concurrent
from concurrent.futures import ProcessPoolExecutor, as_completed

from divercity_utils import parallel_compute_divercity_score_weighted
from results_utils import prepare_flatten_dict, get_divercity_vs_radius_at_k,filter_by_p_eps, get_divercity_at_k




def create_gridded_graph(center=(0, 0), rows=5, cols=5, row_size=1, col_size=1, edge_cost=1):
    """
    Create a gridded directed weighted graph with specified parameters.

    Parameters:
    - center: Tuple of (x, y) coordinates for the center of the grid.
    - rows: Number of rows in the grid.
    - cols: Number of columns in the grid.
    - row_size: Distance between adjacent rows.
    - col_size: Distance between adjacent columns.
    - edge_cost: Cost for each edge in the graph.

    Returns:
    - G: A directed weighted graph.
    """
    G = nx.DiGraph()
    cx, cy = center

    rows += 1
    cols += 1
    
    # Calculate the offset to ensure (0, 0) is at the center
    row_offset = -(rows // 2)
    col_offset = -(cols // 2)

    # Add nodes with attributes
    for r in range(rows):
        for c in range(cols):
            x = cx + (col_offset + c) * col_size
            y = cy + (row_offset + r) * row_size
            G.add_node((r, c), x=x, y=y)

    # Add edges with costs
    for r in range(rows):
        for c in range(cols):
            current_node = (r, c)
            if r < rows - 1:  # Connect to the node below
                G.add_edge(current_node, (r + 1, c), cost=edge_cost)
            if c < cols - 1:  # Connect to the node to the right
                G.add_edge(current_node, (r, c + 1), cost=edge_cost)
            if r > 0:  # Connect to the node above
                G.add_edge(current_node, (r - 1, c), cost=edge_cost)
            if c > 0:  # Connect to the node to the left
                G.add_edge(current_node, (r, c - 1), cost=edge_cost)

    return G


def fixed_radius_sampling_grid(center, radius, nb_samples):
    """
    Generate sample points on a circle in Cartesian coordinates.
    """
    cx, cy = center
    theta = np.linspace(0, 2 * np.pi, nb_samples, endpoint=False)
    points = [(cx + radius * np.cos(t), cy + radius * np.sin(t)) for t in theta]
    return points


def perform_sampling_grid(G, r_list, city_center, n_samples_circle, max_dist=-1):#, kd_tree):
    
    # (x, y) representing the sampled points
    sampled_points = {}

    # closest graph node associated with the sampled points
    sampled_nodes = {}

    # sampled nodes coordinates
    sampled_nodes_coordinates = {}

    # Create a KD-tree
    list_node_ids_kdt = list(G.nodes())
    node_coordinates = [(data['x'], data['y']) for _, data in G.nodes(data=True)]
    kd_tree = cKDTree(node_coordinates)
        
    for r in r_list:

        points_r_km = fixed_radius_sampling_grid(city_center, r, n_samples_circle)
        sampled_points[r] = points_r_km

        nodes_r_km = []

        for (x, y) in points_r_km:
            # Query the closest graph node
            dist, nearest_node_index = kd_tree.query((x, y))
            node_id_nearest_node = list_node_ids_kdt[nearest_node_index]

            node_data = G.nodes[node_id_nearest_node]
            x_node = node_data["x"]
            y_node = node_data["y"]

            if max_dist > 0:

                if dist <= max_dist:
                    nodes_r_km.append(node_id_nearest_node)
                    sampled_nodes_coordinates[node_id_nearest_node] = (x, y)
                
            else:
                nodes_r_km.append(node_id_nearest_node)
                sampled_nodes_coordinates[node_id_nearest_node] = (x, y)
            

        sampled_nodes[r] = nodes_r_km
        
        
        sampling_info = {}

        sampling_info["sampling_parameters"] = {"r_list": list(r_list), 
                                                "n_samples_circle": n_samples_circle}

        sampling_info["sampled_points"] = sampled_points        
        sampling_info["sampled_nodes"] = sampled_nodes
        sampling_info["sampled_nodes_coordinates"] = sampled_nodes_coordinates

        
    return sampling_info


def select_edges_in_square(G, center, side_length):
    """
    Select edges forming the boundary of a square centered at 'center' with side length 'side_length'.
    """
    cx, cy = center
    half_side = side_length / 2

    # Define the boundary conditions
    def is_on_boundary(x, y):
        return (abs(x - cx) == half_side and -half_side <= y - cy <= half_side) or \
               (abs(y - cy) == half_side and -half_side <= x - cx <= half_side)

    # Select edges where both nodes are on the boundary and adjacent
    return [
        edge for edge in G.edges
        if is_on_boundary(G.nodes[edge[0]]['x'], G.nodes[edge[0]]['y']) and
           is_on_boundary(G.nodes[edge[1]]['x'], G.nodes[edge[1]]['y'])
    ]


def select_edges_in_row(G, y_value):
    """
    Select edges in a specific row based on the y-coordinate value.

    Parameters:
    - G: The graph.
    - y_value: The y-coordinate value of the row.

    Returns:
    - List of edge tuples in the specified row.
    """
    def is_in_row(y):
        return y == y_value

    # Select edges where both nodes lie in the specified row
    return [
        edge for edge in G.edges
        if is_in_row(G.nodes[edge[0]]['y']) and is_in_row(G.nodes[edge[1]]['y'])
    ]


def select_edges_in_row_lim(G, y_value, col_min, col_max):
    """
    Select edges in a specific row based on the y-coordinate value and x-coordinate range.

    Parameters:
    - G: The graph.
    - y_value: The y-coordinate value of the row.
    - col_min: The minimum x-coordinate value (inclusive).
    - col_max: The maximum x-coordinate value (inclusive).

    Returns:
    - List of edge tuples in the specified row and column range.
    """
    def is_in_row(y):
        return y == y_value

    def is_in_column_range(x):
        return col_min <= x <= col_max

    # Select edges where both nodes lie in the specified row and column range
    return [
        edge for edge in G.edges
        if is_in_row(G.nodes[edge[0]]['y']) and is_in_row(G.nodes[edge[1]]['y']) and
           is_in_column_range(G.nodes[edge[0]]['x']) and is_in_column_range(G.nodes[edge[1]]['x'])
    ]



def select_edges_in_column(G, x_value):
    """
    Select edges in a specific column based on the x-coordinate value.

    Parameters:
    - G: The graph.
    - x_value: The x-coordinate value of the column.

    Returns:
    - List of edge tuples in the specified column.
    """
    def is_in_column(x):
        return x == x_value

    # Select edges where both nodes lie in the specified column
    return [
        edge for edge in G.edges
        if is_in_column(G.nodes[edge[0]]['x']) and is_in_column(G.nodes[edge[1]]['x'])
    ]



def select_edges_in_circle(G, center, radius, tolerance=1e-6):
    """
    Select edges forming the boundary of a circle centered at 'center' with a radius 'radius'.
    
    Parameters:
    - G: The graph with nodes having 'x' and 'y' attributes.
    - center: Tuple (cx, cy) specifying the circle's center.
    - radius: The radius of the circle.
    - tolerance: Allowed deviation for a node to be considered on the circle's boundary.
    
    Returns:
    - List of edges where both nodes lie on the circle's boundary.
    """
    cx, cy = center

    # Define a function to check if a point lies on the circle's boundary (within tolerance)
    def is_on_circle(x, y):
        return abs((x - cx)**2 + (y - cy)**2 - radius**2) <= tolerance

    # Select edges where both nodes are on the circle's boundary
    return [
        edge for edge in G.edges
        if is_on_circle(G.nodes[edge[0]]['x'], G.nodes[edge[0]]['y']) and
           is_on_circle(G.nodes[edge[1]]['x'], G.nodes[edge[1]]['y'])
    ]



def distance_nodes_attractors(node_list, G):

    merged_attractors = [item for sublist in G.graph["attractors"] for item in sublist]
    edge_geometries = [LineString([(G.nodes[u]['x'], G.nodes[u]['y']),
                                   (G.nodes[v]['x'], G.nodes[v]['y'])]) for u, v in merged_attractors]
    
    edge_gdf = gpd.GeoDataFrame([{'source': u, 'target': v} for u, v in merged_attractors], geometry=edge_geometries)
    edges = edge_gdf['geometry'].tolist()
    
    list_distances = [distance_to_closest_edge(node, edges) for node in node_list]

    return list_distances


def distance_to_closest_edge(node, edges):

    node_point = Point(node)  # Convert the node to a Shapely Point object
    min_distance = float('inf')  # Initialize with a large value

    for edge in edges:
        distance = node_point.distance(edge)  # Calculate distance to the edge
        if distance < min_distance:
            min_distance = distance  # Update the minimum distance if a closer edge is found

    return min_distance



def launch_simplified_model(k, list_p, list_eps, r_list, n_samples,
                  center=(0,0), n_rows=40, n_cols=40, cell_size_km=0.5,
                  edge_speed_kmh = 50, edge_attractor_speed_kmh = 100,
                  km_sq_attractors=[],
                  rows_attractors=[],
                  cols_attractors=[],
                  water_columns=[],
                  bridges_at_rows=[],
                  max_it=1000, njobs=10):

    """
    Launch a simplified model of route selection and diversification
    on a synthetic grid-based transportation network.

    This function creates a 2D gridded graph representing a simplified city,
    optionally including attractors (areas or corridors of lower travel cost),
    water bodies (impassable regions), and bridges (crossings with lower cost).
    It then simulates navigation behavior through path penalization and computes
    DiverCity metrics for the sampled origin-destination pairs.

    Parameters
    ----------
    k : int
        Maximum number of alternative paths considered in DiverCity computation.
    list_p : list of float
        Penalization parameters applied during path generation (route conformity simulation).
    list_eps : list of float
        Tolerance parameters used when computing DiverCity similarity scores.
    r_list : list of float
        List of radii (in km) from the city center used to sample origins and destinations.
    n_samples : int
        Number of origin-destination samples per radius.
    center : tuple of float, optional
        Coordinates (x, y) for the network center. Default is (0, 0).
    n_rows : int, optional
        Number of rows in the grid network. Default is 40.
    n_cols : int, optional
        Number of columns in the grid network. Default is 40.
    cell_size_km : float, optional
        Physical size (km) of each grid cell. Default is 0.5 km.
    edge_speed_kmh : float, optional
        Default speed (km/h) on regular edges. Used to compute base travel cost.
    edge_attractor_speed_kmh : float, optional
        Speed (km/h) on attractor edges (e.g., main roads, bridges).
    km_sq_attractors : list of float, optional
        Radii (km) defining square-shaped attractor zones centered on the city.
    rows_attractors : list of float, optional
        Row indices (or row distances in km) where entire horizontal corridors act as attractors.
    cols_attractors : list of float, optional
        Column indices (or column distances in km) where entire vertical corridors act as attractors.
    water_columns : list of int, optional
        Column indices that represent impassable water bodies (e.g., a river or sea inlet).
    bridges_at_rows : list of int, optional
        Row indices where bridge crossings are allowed across water columns.
    max_it : int, optional
        Maximum number of iterations for path penalization algorithm.
    njobs : int, optional  
    Number of parallel jobs to use for computation.

    Returns
    -------
    G : networkx.Graph
        The processed grid graph (with removed water nodes and attractor costs).
    df : pandas.DataFrame
        Summary DataFrame containing DiverCity metrics for each OD pair and k value.
    penalized_paths_dict : dict
        Dictionary containing penalized paths computed for each OD pair
    """
    
    print("\n" + "="*40)
    print(f"         Initializing Simplified Model [njobs = {njobs}]         ")
    print("="*40)

    # Convert speed to travel time
    edge_cost = (1000 * cell_size_km) / (edge_speed_kmh / 3.6)
    edge_attractor_cost = (1000 * cell_size_km) / (edge_attractor_speed_kmh / 3.6)
    
    print(f"Grid Parameters: {n_rows} x {n_cols}, Cell Size: {cell_size_km} km")
    print(f"Edge Speed: {edge_speed_kmh} km/h, Edge Travel Time Cost: {edge_cost:.2f}")
    print(f"Attractor Edge Speed: {edge_attractor_speed_kmh} km/h, Attractor Travel Time Cost: {edge_attractor_cost:.2f}")
    print("-" * 40)

    # Create grid-based graph
    G = create_gridded_graph(center=center, rows=n_rows, cols=n_cols, row_size=cell_size_km, col_size=cell_size_km, edge_cost=edge_cost)
    G.graph["attractors"] = []
    
    print("Grid-based graph created successfully.")

    # Bridges processing
    nodes_bridge = []
    tot_edges_bridge = []
    
    if bridges_at_rows and water_columns:
        print(f"Processing {len(bridges_at_rows)} bridges...")
        for bridge_at_row in bridges_at_rows:
            edges_bridge = select_edges_in_row_lim(G, bridge_at_row, np.min(water_columns), np.max(water_columns))
            set_row_bridges = set(edges_bridge)
            tot_edges_bridge += list(edges_bridge)
            nodes_bridge += [w_edge[0] for w_edge in set_row_bridges]

        print(f"Total bridges processed: {len(tot_edges_bridge)}")

    # Water body processing
    if water_columns:
        print(f"Processing water columns: {water_columns}")
        tot_removed_nodes = 0
        for w_column in water_columns:
            water_edges = select_edges_in_column(G, w_column)
            nodes_water = {w_edge[i] for w_edge in water_edges for i in [0,1] if w_edge[i] not in nodes_bridge}
            tot_removed_nodes += len(nodes_water)
            G.remove_nodes_from(nodes_water)

        print(f"Total water nodes removed: {tot_removed_nodes}")

    # Apply bridge attractor cost
    for edge in tot_edges_bridge:
        G[edge[0]][edge[1]]['cost'] = edge_attractor_cost
    G.graph["attractors"].append(tot_edges_bridge)

    # Process different types of attractors
    print(f"Processing attractors: {len(km_sq_attractors) + len(rows_attractors) + len(cols_attractors)} in total.")
    print(f"Attractors placed at {km_sq_attractors + rows_attractors + cols_attractors} km from the city center.")


    for r_attractor in km_sq_attractors:
        square_ring = select_edges_in_square(G, center, r_attractor * 2)
        for edge in square_ring:
            G[edge[0]][edge[1]]['cost'] = edge_attractor_cost
        G.graph["attractors"].append(square_ring)

    for row_attr_km in rows_attractors:
        attr_row = select_edges_in_row(G, row_attr_km)
        for edge in attr_row:
            G[edge[0]][edge[1]]['cost'] = edge_attractor_cost
        G.graph["attractors"].append(attr_row)

    for col_attr_km in cols_attractors:
        attr_col = select_edges_in_column(G, col_attr_km)
        for edge in attr_col:
            G[edge[0]][edge[1]]['cost'] = edge_attractor_cost
        G.graph["attractors"].append(attr_col)

    print("Attractors processed successfully.")

    # Perform grid sampling
    print("\n" + "="*40)
    print("         Performing Grid Sampling         ")
    print("="*40)

    if water_columns:
        dict_sampling = perform_sampling_grid(G, r_list, center, n_samples, max_dist=1)
    else:
        dict_sampling = perform_sampling_grid(G, r_list, center, n_samples)

    sampled_nodes = dict_sampling["sampled_nodes"]
    print(f"Total nodes sampled: {len(sampled_nodes)}")

    # Convert to igraph
    G_ig = ig.Graph.from_networkx(G)

    node_nx_to_ig = {n["_nx_name"]: ind_n for ind_n, n in enumerate(G_ig.vs())}
    node_ig_to_nx = {ind_n: n["_nx_name"] for ind_n, n in enumerate(G_ig.vs())}
    
    G_ig["info"] = {"node_nx_to_ig": node_nx_to_ig, "node_ig_to_nx": node_ig_to_nx}

    print("\n" + "="*40)
    print("         Computing Path Penalization         ")
    print("="*40)

    
    manager = multiprocessing.Manager()
    penalized_paths_dict = manager.dict()

    with ProcessPoolExecutor(max_workers=njobs) as executor:
        futures = {
            executor.submit(
                compute_path_penalization_r,
                G_ig, r, list_p, k, sampled_nodes,
                penalized_paths_dict, "cost", max_it
            ): r for r in sampled_nodes.keys()
        }

        for future in as_completed(futures):
            r = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error computing penalized paths for radius {r}: {e}")

    print("Path penalization computation completed.")

    penalized_paths_dict = dict(penalized_paths_dict)

    print("\n" + "="*40)
    print("         Computing DiverCity Scores         ")
    print("="*40)

    edge_lengths = {(edge_data[0], edge_data[1]): 1 for edge_data in G.edges(data=True)}
    
    manager = multiprocessing.Manager()
    results_dict = manager.dict()

    with ProcessPoolExecutor(max_workers=njobs) as executor:
        futures = {
            executor.submit(
                parallel_compute_divercity_score_weighted,
                sampled_nodes, r, k, penalized_paths_dict,
                n_samples, list_p, list_eps, edge_lengths, results_dict
            ): r for r in sampled_nodes.keys()
        }

        for future in as_completed(futures):
            r = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error computing DiverCity for radius {r}: {e}")

    results_dict = dict(results_dict)

    print("DiverCity scores computation completed.")

    print("\n" + "="*40)
    print("         Preparing Results DataFrame         ")
    print("="*40)

    flattened_data = prepare_flatten_dict(results_dict, dict_sampling, max_k=k)
    df = pd.DataFrame(flattened_data)
    
    for at_k in np.arange(2, k+1):
        df[f"avg_jaccard_at_k_{at_k}"].fillna(1, inplace=True)
        df[f"div_k_{at_k}"] = (1 - df[f"avg_jaccard_at_k_{at_k}"]) * df[f"number_nsp_at_k_{at_k}"]

    print("DataFrame prepared successfully.")
    
    return G, df, penalized_paths_dict



def plot_grid_map(G, linewidth_attr=5):

    Gplot = G.to_undirected()

    node_geometries = [Point(data['x'], data['y']) for _, data in Gplot.nodes(data=True)]
    node_gdf = gpd.GeoDataFrame([{'id': n, **data} for n, data in Gplot.nodes(data=True)], geometry=node_geometries)
    
    # Create a GeoDataFrame for the edges
    edge_geometries = [LineString([(Gplot.nodes[u]['x'], Gplot.nodes[u]['y']),
                                   (Gplot.nodes[v]['x'], Gplot.nodes[v]['y'])]) for u, v in Gplot.edges()]
    
    edge_gdf = gpd.GeoDataFrame([{'source': u, 'target': v} for u, v in Gplot.edges()], geometry=edge_geometries)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    edge_gdf.plot(linewidth=.1, ax=ax)

    if "attractors" in Gplot.graph:
    
        for attractors in Gplot.graph["attractors"]:
            filtered_edge_gdf = edge_gdf[edge_gdf.apply(lambda row: (row['source'], row['target']) in attractors or 
                                    (row['target'], row['source']) in attractors, axis=1)]
        
            filtered_edge_gdf.plot(ax=ax, linewidth=linewidth_attr)
    
    ax.scatter(0,0, marker="o", c="k")

    return ax
    


