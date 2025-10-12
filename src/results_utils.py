import os
import pandas as pd
import json
import numpy as np
import osmnx as ox
import math

def process_city(city, result_folder, max_k=15):
    return city, prepare_df_results(result_folder, max_k=max_k)


def prepare_flatten_dict(dict_results, dict_sampling, max_k=15):

    # Flatten the structure
    flattened_data = []
    
    for r_key, r_value in dict_results.items():
        for od_key, od_value in r_value.items():
            for eps_key, eps_value in od_value.items():
                for p_key, scores in eps_value.items():
                    # Create a record for each 'leaf' with its hierarchy as separate columns
                    origin_idx = int(od_key.split("_")[0])
                    dest_idx = int(od_key.split("_")[1])
    
                    origin_node = dict_sampling["sampled_nodes"][r_key][origin_idx]
                    dest_node = dict_sampling["sampled_nodes"][r_key][dest_idx]


                    dict_row_info = {'r': float(r_key),
                                'origin_idx': origin_idx,
                                'dest_idx': dest_idx,
                                'origin_node': origin_node,
                                'dest_node': dest_node,
                                'p': float(p_key),
                                'eps': float(eps_key),
                                'edge_weight': "travel_time",
                                'avg_jaccard': scores["avg_jaccard"],
                                'number_paths': scores["number_paths"],
                                'number_nsp': scores["number_nsp"],
                                'tot_its': scores["list_iterations"][-1],
                                'divercity': scores["divercity"]}

                    for first_k in np.arange(2, max_k+1):
                        dict_row_info[f"number_nsp_at_k_{first_k}"] = scores[f"number_nsp_at_k_{first_k}"]
                        dict_row_info[f"avg_jaccard_at_k_{first_k}"] = scores[f"avg_jaccard_at_k_{first_k}"]
        
                    flattened_data.append(dict_row_info)

    return flattened_data


def prepare_df_results(result_folder, max_k=15):

    # load sampling infos:
    with open(f'{result_folder}sampling_info.json', 'r') as file:
        dict_sampling = json.load(file)

    # load results
    with open(f'{result_folder}results.json', 'r') as file:
        dict_results = json.load(file)

    # flatten data
    flattened_data = prepare_flatten_dict(dict_results, dict_sampling, max_k=max_k)

    # create DataFrame
    df = pd.DataFrame(flattened_data)

    # add extra info (e.g., od-distance)
    list_od_distance = []
    
    for origin_node, dest_node in zip(df["origin_node"], df["dest_node"]):
        
        lng_o, lat_o = dict_sampling["sampled_nodes_coordinates"][str(origin_node)]
        lng_d, lat_d = dict_sampling["sampled_nodes_coordinates"][str(dest_node)]
        
        haversine_distance_km = ox.distance.great_circle(lat_o, lng_o, lat_d, lng_d)/1e3
        list_od_distance.append(haversine_distance_km)
    
    df["od_dist_km"] = list_od_distance

    for at_k in np.arange(2, max_k+1):
        df[f"avg_jaccard_at_k_{at_k}"].fillna(1, inplace=True)
        df[f"div_k_{at_k}"] = (1-df[f"avg_jaccard_at_k_{at_k}"])*df[f"number_nsp_at_k_{at_k}"]

    return df



def filter_by_p_eps(df, p, eps):
    df_p_eps = df[(df["p"]==p)&(df["eps"]==eps)]
    return df_p_eps

def get_divercity_at_k(df, at_k, p, eps):

    df_p_eps = filter_by_p_eps(df, p, eps)
    #df_p_eps[f"avg_jaccard_at_k_{at_k}"].fillna(1, inplace=True)
    #df_p_eps[f"div_k_{at_k}"] = (1-df_p_eps[f"avg_jaccard_at_k_{at_k}"])*df_p_eps[f"number_nsp_at_k_{at_k}"]

    return list(df_p_eps[f"div_k_{at_k}"])


def get_divercity_vs_radius_at_k(df, at_k, p, eps):

    df_p_eps = filter_by_p_eps(df, p, eps)
    #df_p_eps[f"avg_jaccard_at_k_{at_k}"].fillna(1, inplace=True)
    #df_p_eps[f"div_k_{at_k}"] = (1-df_p_eps[f"avg_jaccard_at_k_{at_k}"])*df_p_eps[f"number_nsp_at_k_{at_k}"]
    divercity_vs_radius = list(df_p_eps.groupby(["r"], as_index=False).median().sort_values("r")[f"div_k_{at_k}"])

    return divercity_vs_radius


def get_list_all_and_avg_tt(dict_tt, eps=.3):

    list_all_costs = []
    list_trip_avg_costs = []
    
    for r in dict_tt:
        for od in dict_tt[r]:
            min_c = np.min(dict_tt[r][od])
            NS_cost = [c for c in dict_tt[r][od] if c <= min_c*(1+eps)]
            
            list_all_costs+=NS_cost
            
            avg_NS_cost = np.mean(NS_cost)  
            list_trip_avg_costs.append(avg_NS_cost)

    return list_all_costs, list_trip_avg_costs



def load_network_measures(list_cities, base_path="../data/road_networks_info"):
    """Load and compute total attractor lengths per city."""
    dict_net_measures = {}
    city_to_attr_length = {}

    for city in list_cities:
        with open(f"{base_path}/{city}/info_{city}.json", 'r') as file:
            dict_net_measures[city] = json.load(file)

        edges_of_attractors = dict_net_measures[city]["info_attractors"]["all_edges"]
        dict_info_edges = dict_net_measures[city]["info_edges"]

        total_weights = [
            dict_info_edges[f"{vo}_{vu}"]["length"] / 1e3
            for (vo, vu) in edges_of_attractors
            if dict_info_edges[f"{vo}_{vu}"]["distance_to_center_km"] <= 32
        ]
        city_to_attr_length[city] = sum(total_weights)

    return city_to_attr_length


def load_attractor_dispersion(list_cities, path_disp="../data/attractor_spatial_dispersion", city_to_attr_length=None):
    """Load attractor dispersion info and compute density and homogeneity metrics."""
    dict_attractors_H = {}
    city_to_attr_density = {}
    city_to_attr_land_ratio = {}
    city_to_attr_homo = {}

    for city in list_cities:
        with open(f"{path_disp}/{city}/attractor_spatial_dispersion_{city}.json", 'r') as file:
            dict_attractors_H[city] = json.load(file)

        n_valid = dict_attractors_H[city]["20000"]["1000"]["n_valid_points"]
        n_invalid = dict_attractors_H[city]["20000"]["1000"]["n_invalid_points"]
        land_ratio = n_valid / (n_valid + n_invalid)
        city_to_attr_land_ratio[city] = land_ratio

        # Compute density (total attractor length / land area)
        A = np.pi * 32**2
        total_distance = city_to_attr_length[city]
        city_to_attr_density[city] = total_distance / (land_ratio * A)

        # Homogeneity = mean distance of random points to attractors (H)
        list_distances = dict_attractors_H[city]["20000"]["1000"]["distances"]
        city_to_attr_homo[city] = np.mean(list_distances)

    return city_to_attr_density, city_to_attr_homo


def load_distance_node_attractors(list_cities, base_path="../data/distance_node_to_attractors"):
    """Load distance-to-attractor dictionaries per city."""
    dict_distance_node_attractors = {}
    for city in list_cities:
        with open(f"{base_path}/{city}/distance_node_to_attractors_{city}.json", 'r') as file:
            dict_distance_node_attractors[city] = json.load(file)
    return dict_distance_node_attractors


# ============================================================
# 2. COMPUTATION HELPERS
# ============================================================

def compute_node_divercity_table(list_cities, dict_df_results, dict_distance_node_attractors, p, eps, at_k, min_radius=2):
    """Compute node-level DiverCity and distance to attractors for each city."""
    list_percentiles = np.arange(10, 101, 10)

    LIST_r, LIST_node, LIST_node_div, LIST_node_dist_A = [], [], [], []
    LIST_city, LIST_avg_dist_city = [], []
    dict_LIST_percentile_city = {pc: [] for pc in list_percentiles}

    for city in list_cities:
        df = dict_df_results[city]
        dict_node_attr = dict_distance_node_attractors[city]

        df_p_eps = filter_by_p_eps(df, p, eps)
        df_p_eps = df_p_eps[df_p_eps["r"] > min_radius]

        tmp_list_div_nodes = []
        tmp_list_dist_nodes = []

        for r in np.arange(min_radius + 1, 31, 1):
            df_r = df_p_eps[df_p_eps["r"] == r]
            nodes_in_r = list(set(df_r["origin_node"]).union(df_r["dest_node"]))

            for node in nodes_in_r:
                node_div = df_r[
                    (df_r["origin_node"] == node) | (df_r["dest_node"] == node)
                ][f"div_k_{at_k}"].mean()

                tmp_list_div_nodes.append(node_div)
                tmp_list_dist_nodes.append(dict_node_attr[str(node)])

                LIST_r.append(r)
                LIST_node.append(node)
                LIST_node_div.append(node_div)
                LIST_node_dist_A.append(dict_node_attr[str(node)])

        LIST_city += [city] * len(tmp_list_div_nodes)
        LIST_avg_dist_city += [np.mean(tmp_list_dist_nodes)] * len(tmp_list_dist_nodes)

        for pc in list_percentiles:
            dict_LIST_percentile_city[pc] += [np.percentile(tmp_list_div_nodes, pc)] * len(tmp_list_div_nodes)

    df_node_dcity = pd.DataFrame({
        "city": LIST_city,
        "node": LIST_node,
        "r": LIST_r,
        f"div_k_{at_k}": LIST_node_div,
        "dist_attr": LIST_node_dist_A,
        "city_dist": LIST_avg_dist_city,
    })

    for pc in list_percentiles:
        df_node_dcity[f"pct_{pc}"] = dict_LIST_percentile_city[pc]

    return df_node_dcity, list_percentiles


def compute_city_median_divercity(dict_df_results, list_cities, k, p, eps):
    """
    Compute the median DiverCity value for each city.
    
    Parameters
    ----------
    dict_df_results : dict
        Dictionary mapping city names to their corresponding results DataFrame.
    list_cities : list
        List of city names.
    k, p, eps : numeric
        Parameters for DiverCity computation.
        
    Returns
    -------
    dict
        Dictionary mapping city -> median DiverCity value.
    dict
        Dictionary mapping city -> IQR DiverCity value.
    """
    city_to_median_dc = {}
    city_to_iqr_dc = {}

    for city in list_cities:
        df = dict_df_results[city]
        y_vector = get_divercity_at_k(df, k, p, eps)
        city_to_median_dc[city] = np.median(y_vector)
        city_to_iqr_dc[city] = np.percentile(y_vector, 75) - np.percentile(y_vector, 25)
    
    return city_to_median_dc, city_to_iqr_dc


def load_speed_scenarios(list_cities, speeds_to_load, exp_id, at_k, base_path="../data/results"):
    """
    Load DiverCity results for each city and speed reduction scenario.
    Returns dict_df_speed[speed][city] = dataframe.
    """
    dict_df_speed = {s: {} for s in speeds_to_load}
    for s in speeds_to_load:
        for city in list_cities:
            city_name, df = process_city(
                f"{city}_speed{s}",
                f"{base_path}/{city}_speed{s}_{exp_id}/",
                max_k=at_k
            )
            dict_df_speed[s][city_name] = df
    return dict_df_speed


def load_travel_times(list_cities, p, base_path="../data/route_measures"):
    """Load precomputed travel time dictionaries for each city."""
    dict_travel_times = {}
    for city in list_cities:
        with open(f"{base_path}/travel_time_{city}_{str(p).replace('.','p')}.json", 'r') as file:
            dict_travel_times[city] = json.load(file)
    return dict_travel_times


def compute_divercity_gain(dict_df_results, dict_df_speed, list_cities, speeds_to_load, at_k, p, eps):
    """Compute median DiverCity gains under speed limit reductions."""
    dict_gains = {}
    for city in list_cities:
        dict_gains[city] = {"median": []}
        df_base = filter_by_p_eps(dict_df_results[city], p, eps)
        median_full = np.median(get_divercity_at_k(df_base, at_k, p, eps))
        for s in speeds_to_load:
            df_speed = dict_df_speed[s][f"{city}_speed{s}"]
            y_vector = get_divercity_at_k(df_speed, at_k, p, eps)
            dict_gains[city]["median"].append(np.median(y_vector) - median_full)
    return dict_gains


def compute_travel_time_difference(dict_travel_times, list_cities, speeds_to_load):
    """Compute average travel time differences (in seconds) under speed limit reductions."""
    dict_tt_difference = {}
    for city in list_cities:
        dict_tt_difference[city] = {"trip": []}

        d_original = dict_travel_times[city]["original"]
        all_tt_original, avg_tt_original = get_list_all_and_avg_tt(d_original, eps=0.3)
        avg_tt_original = np.mean([x for x in avg_tt_original if not math.isinf(x)])

        for s in speeds_to_load:
            d_speed = dict_travel_times[city][f"{s}"]
            all_tt_speed, avg_tt_speed = get_list_all_and_avg_tt(d_speed, eps=0.3)
            avg_tt_speed = np.mean([x for x in avg_tt_speed if not math.isinf(x)])
            dict_tt_difference[city]["trip"].append(avg_tt_speed - avg_tt_original)

    return dict_tt_difference



