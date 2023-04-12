import json
import csv

def csv_to_json(csv_file_path, json_file_path):
    '''reads data from a CSV file stored at csv_file_path and 
    converts the csv into a json file which is stored at json_file_path'''
    
    temp = {} # used to store key-value pairs for each row of data
    keys = [] # stores the keys of the CSV file
    out = [] # will contain a list of dictionaries 

    with open(csv_file_path, newline ='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in enumerate (reader):
            if (row[0] == 0):
                keys = list(row[1].keys())
                # getting the keys from the csv
                keys.pop(0)
            
            for key in keys:
                # storing each line of data in temp
                temp[key] = row[1][key]

            # adding each row to out
            out.append(temp.copy())

    # creating the json file
    jsonStr = json.dumps(out, indent=4)

    # writing json to specified path
    with open(json_file_path, 'w') as json_file:
        json_file.write(jsonStr)
