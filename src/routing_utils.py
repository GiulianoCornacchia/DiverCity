import igraph
import numpy as np
import warnings


def route_cost_nx(G, route_nx, attribute):
    return sum(G[u][v][0][attribute] for u, v in zip(route_nx[:-1], route_nx[1:]))
    

def node_to_edge_list_ig(G, route):
    return [G.get_eid(route[i], route[i + 1]) for i in range(len(route) - 1)]
    

def route_cost_ig(G, route, attribute):

    path_edges = node_to_edge_list_ig(G, route)

    # Sum the weights of these edges
    route_cost = sum(G.es[path_edges][attribute])

    return route_cost


def check_if_connected(G, node_src, node_dest):

    dict_nx_to_ig = G["info"]["node_nx_to_ig"]

    node_src_ig = dict_nx_to_ig[node_src]
    node_dest_ig = dict_nx_to_ig[node_dest]

    return len(G.get_shortest_paths(node_src_ig, node_dest_ig, mode="OUT", output="vpath")[0])>0




def path_penalization(G_ig, node_src, node_dest, k, p, attribute, all_distinct=True, remove_tmp_attribute=True, max_iter=1e3):

    # dict_ig_to_nx dict to map IG ids back into the "original" ones (that is NetworkX)
    
    # Initialize iteration counter and result containers
    it = 0
    result_list, path_list = [], []
    
    # Create a temporary copy of the edge attribute to penalize
    G_ig.es[f"tmp_{attribute}"] = G_ig.es[attribute]

    dict_ig_to_nx = G_ig["info"]["node_ig_to_nx"]
    dict_nx_to_ig = G_ig["info"]["node_nx_to_ig"]

    node_src_ig = dict_nx_to_ig[node_src]
    node_dest_ig = dict_nx_to_ig[node_dest]
    
    # Iterate to find k distinct paths or until max_iter is reached
    while len(result_list) < k and it < max_iter:
    
        # Compute the shortest path using the temporary penalized attribute
        sp_k = G_ig.get_shortest_paths(node_src_ig, node_dest_ig, weights=f"tmp_{attribute}", mode="OUT", output="vpath")[0];
    
        # If the path is distinct (or if duplicates are allowed), add it to the result list
        if not (all_distinct and sp_k in path_list):
            
            # Compute the original cost
            original_cost = route_cost_ig(G_ig, sp_k, attribute)
        
            # Compute the penalized cost
            penalized_cost = route_cost_ig(G_ig, sp_k, f"tmp_{attribute}")
                
            # Store the path details in the result list
            # store the route as sequence of edges
            result_list.append({"node_list_nx": [dict_ig_to_nx[v] for v in sp_k], # translate into networkx vertices
                                    "original_cost": original_cost,
                                    "penalized_cost": penalized_cost,
                                    "iteration": it})
    
            # Keep track of the path to ensure uniqueness
            path_list.append(sp_k)
    
        # apply the penalization to the current sp's edges
        edge_list_sp_k = node_to_edge_list_ig(G_ig, sp_k)
        for e in G_ig.es(edge_list_sp_k):
            e[f"tmp_{attribute}"] *= (1 + p)
    
        # Increment the iteration counter
        it += 1
    
    # Check if the maximum number of iterations was reached
    #if it == max_iter:
    #    print(f'Iterations limit reached, returned {len(result_list)} distinct paths (instead of {k}).')
    #    warnings.warn(f'Iterations limit reached, returned {len(result_list)} distinct paths (instead of {k}).', RuntimeWarning)
        
    # Remove the temporary attribute from the graph if required
    if remove_tmp_attribute:
        del(G_ig.es[f"tmp_{attribute}"])
        
    return result_list






def compute_path_penalization_r(G, r, list_p, k, sampled_elements, penalized_paths_dict, attribute="travel_time", k_is_route_count=True, max_it=1e3):
    """
    Efficiently compute parallel shortest paths with path penalization for a set of sampled origin-destination (OD) pairs.
    The penalization can be applied for either a fixed number of routes or iterations, depending on the value of `k_is_route_count`.
    Results are stored in a dictionary and can be based on node connectivity or edge attributes.

    Parameters:
    ----------
    G : igraph.Graph
        The graph representing the road network or structure for path computations.
    r : int
        Index for the current sampling iteration. Used to identify the current batch of origin-destination pairs.
    list_p : list
        A list of penalty factors used in the path penalization process. These factors modify the path computation to diversify the routes.
    k : int
        If `k_is_route_count` is True, `k` is the desired number of routes to compute for each origin-destination pair.
        If `k_is_route_count` is False, `k` represents the number of iterations for penalization in the path-finding algorithm.
    sampled_elements : dict
        A dictionary containing the sampled origin and destination points (nodes or objects) for each iteration `r`.
        The dictionary should have the form {r: [origins, destinations]}.
    divercity_dict : dict
        A dictionary to store the results for different origin-destination pairs and penalty factors.
        The format is {iteration: {OD_pair: {penalty_factor: [routes]}}}.
    attribute : str, optional
        The edge attribute to minimize during path computation (default is 'traveltime').
        Other attributes could be distance, cost, etc.
    k_is_route_count : bool, optional
        If True, `k` specifies the number of desired routes to be computed within `max_it` iterations.
        If False, `k` specifies the number of penalization iterations, and the function will return routes after exactly `k` iterations.
        Default is True.
    max_it : int, optional
        Maximum number of iterations to attempt when computing exactly `k` routes (used when `k_is_route_count` is True). Default is 1000.

    Returns:
    -------
    None
        The function modifies `divercity_dict` in place, adding the computed shortest paths and penalized paths for each origin-destination pair.
        Disconnected pairs (those where no valid path exists) are recorded separately.

    Notes:
    -----
    - The function first checks if both source and destination are valid and connected. If not, they are marked as disconnected.
    - The penalization process diversifies the paths by applying the penalty factors from `list_p`.
    - Depending on `k_is_route_count`, the function either computes a fixed number of routes (up to `max_it` iterations) or runs for a fixed number of iterations.
    - The results for each origin-destination pair, including the penalized paths for each penalty factor, are stored in `divercity_dict`.
    
    Example:
    --------
    If you want to compute exactly 5 different routes for each origin-destination pair, use `k=5` and set `k_is_route_count=True`.
    If you want to run 5 iterations of penalization, use `k=5` and set `k_is_route_count=False`.
    """
    
    # Print message indicating the behavior of k
    
    #if k_is_route_count:
    #    print(f"Using k as the number of desired routes (up to {k} routes will be computed in at most {max_it} iterations).")
    #else:
    #    print(f"Using k as the number of iterations (the penalization will run for exactly {k} iterations).")

    dict_ig_to_nx = G["info"]["node_ig_to_nx"]
    dict_nx_to_ig = G["info"]["node_nx_to_ig"]
    
    disconnetted_ods = []  # List to keep track of disconnected origin-destination pairs
    n_points_circle = len(sampled_elements[r])  # Number of points in the sampled circle for this iteration
    dict_tmp = {}  # Temporary dictionary to store results for this iteration

    # Iterate over all possible origin-destination pairs (excluding same-point pairs)
    for ind_src in range(n_points_circle):
        for ind_dest in range(n_points_circle):
            if ind_src != ind_dest:

                # Check if both source and destination are valid nodes (e.g., not in the sea or invalid locations)
                if sampled_elements[r][ind_src] != {} and sampled_elements[r][ind_dest] != {}:
                    
                    # Retrieve source and destination nodes
                    node_from = sampled_elements[r][ind_src]
                    node_to = sampled_elements[r][ind_dest]
                    
                    # Check if the two nodes are connected
                    are_connected = check_if_connected(G, node_from, node_to)
                    
                    if are_connected:
                        # If nodes are connected, initialize a dictionary entry for the OD pair
                        dict_tmp[f"{ind_src}_{ind_dest}"] = {}

                        # For each penalty factor in list_p, compute penalized paths
                        for p in list_p:
                            if k_is_route_count:
                                # Compute exactly `k` routes in at most `max_it` iterations
                                res_pp = path_penalization(G, node_from, node_to, k, p, attribute, max_iter=max_it)
                            else:
                                # Run the path penalization for exactly `k` iterations
                                res_pp = path_penalization(G, node_from, node_to, k, p, attribute, max_iter=k)
                      
                            # Store the penalized paths in the dictionary
                            dict_tmp[f"{ind_src}_{ind_dest}"][p] = res_pp
                            
                    else:
                        # If nodes are not connected, mark this origin-destination pair as disconnected
                        disconnetted_ods.append([r, [ind_src, ind_dest]])

    # Store the results for this sampling iteration in the main divercity dictionary
    penalized_paths_dict[r] = dict_tmp
    
    return





