import lz4.frame
import json
import msgpack
import os
from collections import Counter
from divercity_utils import filter_near_shortest
from my_utils import create_folder_if_not_exists


def compute_traveltime(city, city_speeds, p, exp_id):

    dict_tt_city = {}
    
    with lz4.frame.open(f"../data/results/{city}_{exp_id}/generated_routes.lz4", "rb") as f:
        packed_data = f.read()
        paths_dict_base = msgpack.unpackb(packed_data, strict_map_key=False)
        
    # TRAVEL TIME first k
    dict_tt_city["original"] = {}
    for r in paths_dict_base:
        dict_tt_city["original"][r] = {}
        for od in paths_dict_base[r]:
            list_costs = [alternative["original_cost"] for alternative in paths_dict_base[r][od][p]]
            dict_tt_city["original"][r][od] = list_costs
    
    for city_speed in city_speeds:
        
        with lz4.frame.open(f"../data/results/{city}_speed{city_speed}_{exp_id}/generated_routes.lz4", "rb") as f:
            packed_data = f.read()
            paths_dict_speed = msgpack.unpackb(packed_data, strict_map_key=False)
    
        #speed_name = city_speed.split("_")[-1]

        # TRAVEL TIME first k
        dict_tt_city[city_speed] = {}
        for r in paths_dict_speed:
            dict_tt_city[city_speed][r] = {}
            for od in paths_dict_speed[r]:
                list_costs = [alternative["original_cost"] for alternative in paths_dict_speed[r][od][p]]
                dict_tt_city[city_speed][r][od] = list_costs

                
    create_folder_if_not_exists("../data/route_measures")
    output_file = open(f"../data/route_measures/travel_time_{city}_{str(p).replace('.','p')}.json", "w")
    json.dump(dict_tt_city, output_file)
    output_file.close()

    return 1

