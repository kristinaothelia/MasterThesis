import json
import csv

from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer

# Se paa senere

#json_file = '/itf-fi-ml/home/koolsen/Master/t_data_with_2014nya4.json'
#csv_file = '/itf-fi-ml/home/koolsen/Master/MasterThesis/datasets/t_data_with_2014nya4.csv'

#json_file = '/itf-fi-ml/home/koolsen/Master/t_data_with_2014nya4.json'
#csv_file = '/itf-fi-ml/home/koolsen/Master/MasterThesis/datasets/t_data_with_2014nya4.csv'

path = 'datasets/'

json_R = 'Aurora_R.json'
json_G = 'Aurora_G.json'

csv_R  = 'Aurora_R.csv'
csv_G  = 'Aurora_G.csv'

#import pandas as pd
#df = pd.read_csv (path+csv_R)
#df.to_json(path+json_R)


def csv_to_json(csvFilePath, jsonFilePath):
    jsonArray = []

    #read csv file
    with open(csvFilePath, encoding='utf-8') as csvf:
        #load csv file data using csv library's dictionary reader
        csvReader = csv.DictReader(csvf)

        #convert each csv row into python dict
        for row in csvReader:
            #add this python dict to json array
            jsonArray.append(row)

    #convert python jsonArray to JSON String and write to file
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonString = json.dumps(jsonArray, indent=4)
        jsonf.write(jsonString)

csvFilePath = path+csv_R
jsonFilePath = path+json_R

csv_to_json(csvFilePath, jsonFilePath)
