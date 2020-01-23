import pandas as pd
import json

def populate_features(file_name=None):
    
    file = file_name
    columns = []
    final_dict = {}

    with open(file, 'r') as f:
        json_dict = json.load(f)

    for key, value in json_dict.items():
        if key == "metadata":
            continue
        value_dict = value
        
        for key1, value1 in value_dict.items():
            key_temp = key + "_" + key1
            if type(value1) == type(dict()):
                value1_dict = value1
                
                for key2, value2 in value1_dict.items():
                    if type(value2) == type(dict()) or type(value2) == type(list()):
                        pass
                    else:
                        key_temp = key + "_" + key1+"_"+key2
                        final_dict[key_temp] = value2 
            elif type(value1) != type(list()) and type(value1) != type(str()):
                final_dict[key_temp] = value1 
                
    return final_dict

def get_features(file=None):
    
    file = "data.json"
    columns = []

    with open(file, 'r') as f:
        json_dict = json.load(f)

    for key, value in json_dict.items():
        if key == "metadata":
            continue
        value_dict = value
        
        for key1, value1 in value_dict.items():
            key_temp = key + "_" + key1
            if type(value1) == type(dict()):
                value1_dict = value1
                
                for key2, value2 in value1_dict.items():
                    if type(value2) == type(dict()) or type(value2) == type(list()):
                        pass
                    else:
                        key_temp = key + "_" + key1+"_"+key2
                        columns.append(key_temp)
            elif type(value1) != type(list()) and type(value1) != type(str()):
                columns.append(key_temp)          
    return columns