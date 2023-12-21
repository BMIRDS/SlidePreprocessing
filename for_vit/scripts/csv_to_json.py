import json
import csv

def csv_to_json(csv_file_path, json_file_path, ignore_first_column=True):
    """
    Reads data from a CSV file at the given `csv_file_path`, converts it into a JSON file
    and saves it at the given `json_file_path`.

    Args:
        csv_file_path (str): The path to the CSV file.
        json_file_path (str): The path to the JSON file.

    Returns:
        None.

    Raises:
        FileNotFoundError: If the CSV file cannot be found at the given `csv_file_path`.

    Example Usage:
        csv_to_json('data.csv', 'data.json')
    """
    
    temp = {} # used to store key-value pairs for each row of data
    keys = [] # stores the keys of the CSV file
    out = [] # will contain a list of dictionaries 

    with open(csv_file_path, newline ='') as csvfile:
        reader = csv.DictReader(csvfile)
        keys = reader.fieldnames

        # If the first column should be ignored
        if ignore_first_column:
            keys.pop(0)

        for row in reader:
            temp = {}
            for key in keys:
                value = row[key]
                temp[key] = value
            else:  # This else belongs to the for loop, not the if statement
                out.append(temp)

    # creating the json file
    jsonStr = json.dumps(out, indent=4)

    # writing json to specified path
    with open(json_file_path, 'w') as json_file:
        json_file.write(jsonStr)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file_path', help='The path to the CSV file.')
    parser.add_argument('--json_file_path', help='The path to the JSON file.')
    parser.add_argument('--ignore_first_column', help='Set to True to skip IDs in the first column of the CSV.', default=False, action='store_true')
    args = parser.parse_args()
    csv_to_json(args.csv_file_path, args.json_file_path, args.ignore_first_column)
