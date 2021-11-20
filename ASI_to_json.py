import sys
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import time, datetime
import math

from pathlib import Path
from tqdm import tqdm
from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer
from numba import cuda, jit
#print(cuda.gpus)

# Solar wind parameters we want
SW = {
    0: "Bz, nT (GSE)",
    1: "Bz, nT (GSM)",
    #1: "Speed, km/s",
    #2: "Proton Density, n/cc",
    #3: "Temperature, K"
}

def read_csv(file, print=False):

    data = pd.read_csv(file)
    if print:
        print("From csv file (%s)" %file)
        #data = pd.read_csv(file, index_col=[0])
        print(data)

    return data

#folder = 'JSON_TEST'
#json_file = 'JSON_TEST_TEST.json' # To large fil for GitHub
#csv_file = 'JSON_TEST_TEST.csv'
#wl = 'nan'


omni14 = '.\datasets\omni\omni_min_2014_withDate.csv'
omni16 = '.\datasets\omni\omni_min_2016_withDate.csv'
omni18 = '.\datasets\omni\omni_min_2018_withDate.csv'
omni20 = '.\datasets\omni\omni_min_2020_withDate.csv'

if os.path.isfile(omni14):
    omni_data14_csv = read_csv(file=omni14)
    omni_data16_csv = read_csv(file=omni16)
    omni_data18_csv = read_csv(file=omni18)
    omni_data20_csv = read_csv(file=omni20)
else:
    omni_data14_csv = read_csv(file = '/itf-fi-ml/home/koolsen/Master/MasterThesis/datasets/omni/omni_min_2014_withDate.csv')
    omni_data16_csv = read_csv(file = '/itf-fi-ml/home/koolsen/Master/MasterThesis/datasets/omni/omni_min_2016_withDate.csv')
    omni_data18_csv = read_csv(file = '/itf-fi-ml/home/koolsen/Master/MasterThesis/datasets/omni/omni_min_2018_withDate.csv')
    omni_data20_csv = read_csv(file = '/itf-fi-ml/home/koolsen/Master/MasterThesis/datasets/omni/omni_min_2020_withDate.csv')

# Dataset containing data
def files(green=False, train=False):

    if train:
        folder = '/itf-fi-ml/home/koolsen/Aurora/Data/All_data'
        json_file = '/itf-fi-ml/home/koolsen/Master/Full_aurora_NEW.json' # To large fil for GitHub
        csv_file = '/itf-fi-ml/home/koolsen/Master/MasterThesis/datasets/Full_aurora_NEW.csv'
        wl = '5577 and 6300'

    if green:
        #folder = '/itf-fi-ml/home/koolsen/Master/T_DATA_green'
        #json_file = '/itf-fi-ml/home/koolsen/Master/Aurora_G.json' # To large fil for GitHub
        #csv_file = '/itf-fi-ml/home/koolsen/Master/MasterThesis/datasets/Aurora_G.csv'
        folder = '/itf-fi-ml/home/koolsen/Master/T_DATA_4yr_G'
        json_file = '/itf-fi-ml/home/koolsen/Master/Aurora_4yr_G.json' # To large fil for GitHub
        csv_file = '/itf-fi-ml/home/koolsen/Master/MasterThesis/datasets/Aurora_4yr_G.csv'
        wl = '5577'
    else:
        #folder = '/itf-fi-ml/home/koolsen/Master/T_DATA'
        #json_file = '/itf-fi-ml/home/koolsen/Master/Aurora_R.json' # To large fil for GitHub
        #csv_file = '/itf-fi-ml/home/koolsen/Master/MasterThesis/datasets/Aurora_R.csv'
        folder = '/itf-fi-ml/home/koolsen/Master/T_DATA_4yr_R'
        json_file = '/itf-fi-ml/home/koolsen/Master/Aurora_4yr_R.json' # To large fil for GitHub
        csv_file = '/itf-fi-ml/home/koolsen/Master/MasterThesis/datasets/Aurora_4yr_R.csv'
        wl = '6300'

    return folder, json_file, csv_file, wl

def formats(json_file, csv_file):

    with open(json_file) as json_file:
        data = json.load(json_file)

    df = pd.DataFrame.from_dict(data['entries'])
    #print(df['score'])

    df.to_csv(csv_file, index=False)
    print("json ['entries'] saved as csv file")


@jit()  # nopython=True
def test_new(omni, timepoint):
    for i in range(len(omni)):
        if timepoint in omni[i]:
            return i

def match_dates_omni_aurora_data(omni_data, omni_data_dates, tp, SW):

    index = []
    solarwind = dict()

    ii = test_new(omni_data_dates, tp)
    index.append(ii)
    #print(index)

    #for i in range(len(omni_data['Date'])):
    #    if tp in omni_data['Date'][i]:
    #        index.append(i)

    #print(index)
    #print(omni_data.loc[omni_data.index[index]]["Bz, nT (GSE)"])
    #print(omni_data.loc[omni_data.index[index]]["Bz, nT (GSM)"])
    #print(omni_data.loc[omni_data.index[ii]]["Bz, nT (GSM)"])
    #print(SW[0])

    for k in range(len(SW)):
        #print(SW[k])
        #print(omni_data.loc[omni_data.index[index]][SW[k]].iloc[0])
        solarwind[SW[k]] = omni_data.loc[omni_data.index[index]][SW[k]].iloc[0]


    #Bz_GSE = omni_data.loc[omni_data.index[index]]["Bz, nT (GSE)"]
    #Bz_GSM = omni_data.loc[omni_data.index[index]]["Bz, nT (GSM)"]
    #values.append(Bz_GSE.iloc[0])
    #values.append(Bz_GSM.iloc[0])

    return solarwind

def file_from_ASIfolder(folder, wl, json_file):
    container = DatasetContainer()
    container.from_folder(path=folder,
                          datasetname='Data to be classified',
                          dataset_type='png',
                          wavelength=wl,
                          source='UiO',
                          location='Svaalbard, nya',
                          dataset_description='ASI')

    container.to_json(json_file)
    #formats(json_file, csv_file)

def average_omni_values():
    """
    Use solar wind data from 1 hour before timepoint.
    On night side
    Make 30 min (?) average Bz value (and the others)

    Some std thing?
    """
    print("hmmmmmm..?")

def add_file_information(json_file, csv_file, omni_data, SW, omni=True):

    container = DatasetContainer.from_json(json_file)
    print("length container: ", len(container))

    for entry in tqdm(container):

        path = Path(entry.image_path).stem

        # Add wavelength to json file
        entry.wavelength = container[0]['image_path'][-12:-8]

        date = path.split('_')[1]
        timestamp = path.split('_')[2]
        date = datetime.date(year=int(date[:4]), month=int(date[4:6]), day=int(date[6:]))
        timestamp = datetime.time(hour=int(timestamp[:2]), minute=int(timestamp[2:4]), second=int(timestamp[4:]))
        tiime = datetime.datetime(year=date.year, month=date.month, day=date.day, hour=timestamp.hour, minute=timestamp.minute, second=timestamp.second)

        # Add timepoint to json file
        entry.timepoint = str(tiime)


        if omni:

            # make solar wind data by matchind dates
            tp = entry.timepoint

            if tp[-2:] != "00":
                tp = tp[:-2] + "00"
                #print("30 sec mark, need editing. New time: ", time)

            if tp[:4] == "2014":
                omni_data = omni_data14_csv

            if tp[:4] == "2016":
                omni_data = omni_data16_csv

            if tp[:4] == "2018":
                omni_data = omni_data18_csv

            if tp[:4] == "2020":
                omni_data = omni_data20_csv


            # get only the dates
            omni_data_dates = omni_data['Date']
            omni_data_dates = omni_data_dates.values

            solarwind = match_dates_omni_aurora_data(omni_data, omni_data_dates, tp, SW)
            #print(solarwind)

            # Add solar wind data (dict) to json file
            entry.add_solarwind(solarwind)


    print("Json file updated with additional information")
    print(container)
    container.to_json(json_file)
    formats(json_file, csv_file)

'''
# New training dataset
folder, json_file, csv_file, wl = files(green=False, train=False)
file_from_ASIfolder(folder, wl, json_file)
add_file_information(json_file, csv_file, omni_data20_csv, SW, omni=False)
'''

'''
# red aurora
folder, json_file, csv_file, wl = files(green=False)
#file_from_ASIfolder(folder, wl, json_file)
add_file_information(json_file, csv_file, omni_data20_csv, SW)
'''
# green aurora
folder, json_file, csv_file, wl = files(green=True)
#file_from_ASIfolder(folder, wl, json_file)
add_file_information(json_file, csv_file, omni_data16_csv, SW)
