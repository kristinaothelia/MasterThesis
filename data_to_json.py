from pathlib import Path
import datetime
from tqdm import tqdm
import sys

import pandas as pd
import json

import matplotlib.pyplot as plt

from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer

# Dataset containing data
#folder = r'C:\Users\Krist\Documents\MasterThesis\JSON_TEST'
#folder = r'C:\Users\Krist\Documents\TESTTEST'
#folder = '../T_DATA'

def files(green=False):

    if green:
        folder = '/itf-fi-ml/home/koolsen/Master/T_DATA_green'
        json_file = '/itf-fi-ml/home/koolsen/Master/Aurora_G.json'
        json_slett = '/itf-fi-ml/home/koolsen/Master/MasterThesis/datasets/Aurora_G.json'
        csv_file = '/itf-fi-ml/home/koolsen/Master/MasterThesis/datasets/Aurora_G.csv'
        wl = '5577'
    else:
        folder = '/itf-fi-ml/home/koolsen/Master/T_DATA'
        json_file = '/itf-fi-ml/home/koolsen/Master/Aurora_R.json'
        json_slett = '/itf-fi-ml/home/koolsen/Master/MasterThesis/datasets/Aurora_R.json'
        csv_file = '/itf-fi-ml/home/koolsen/Master/MasterThesis/datasets/Aurora_R.csv'
        wl = '6300'

    return folder, json_file, csv_file, wl

def formats(json_file, csv_file):

    with open(json_file) as json_file:
        data = json.load(json_file)

    df = pd.DataFrame.from_dict(data['entries'])
    #print(df['score'])
    print(df['image_path'])

    df.to_csv(csv_file, index=False)
    print("json ['entries'] saved as csv file")

def make_files(folder, json_file, csv_file, wl):

    container = DatasetContainer()
    container.from_folder(path=folder,
                          datasetname='New data to be classified',
                          dataset_type='png',
                          wavelength=wl,
                          source='UiO',
                          location='Svaalbard, nya',
                          dataset_description='ASI')

    container.to_json(json_file)
    formats(json_file, csv_file)


    container = DatasetContainer.from_json(json_file)
    print("length container: ", len(container))

    for entry in container:
        path = Path(entry.image_path).stem
        entry.wavelength = wl #container[0]['image_path'][-12:-8]
        date = path.split('_')[1]
        timestamp = path.split('_')[2]
        date = datetime.date(year=int(date[:4]), month=int(date[4:6]), day=int(date[6:]))
        timestamp = datetime.time(hour=int(timestamp[:2]), minute=int(timestamp[2:4]), second=int(timestamp[4:]))

        tiime = datetime.datetime(year=date.year, month=date.month, day=date.day, hour=timestamp.hour, minute=timestamp.hour, second=timestamp.second)

        entry.timepoint = str(tiime)

    print("Timepoint updated")
    container = DatasetContainer.from_json(json_file)
    container.to_json(json_file)
    container.to_json(json_slett)
    formats(json_file, csv_file)



folder, json_file, csv_file, wl = files()
make_files(folder, json_file, csv_file, wl)

folder, json_file, csv_file, wl = files(green=True)
make_files(folder, json_file, csv_file, wl)
