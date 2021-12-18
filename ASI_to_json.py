import sys
import pandas as pd
import numpy as np
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

path = '/itf-fi-ml/home/koolsen/Master/'

# Solar wind parameters we want
SW = {
    #0: "Bz, nT (GSE)",
    0: "Bz, nT (GSM)",
    1: "Speed, km/s",
    2: "Proton Density, n/cc",
    #3: "Temperature, K",
}

SW_SD = {
    0: "Bz, nT (GSM), SD",
    1: "Speed, km/s, SD",
    2: "Proton Density, n/cc, SD",
}

def read_csv(file, print=False):

    data = pd.read_csv(file)
    if print:
        print("From csv file (%s)" %file)
        #data = pd.read_csv(file, index_col=[0])
        print(data)

    return data

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
    omni_data14_csv = read_csv(file = path+'MasterThesis/datasets/omni/omni_min_2014_withDate.csv')
    omni_data16_csv = read_csv(file = path+'MasterThesis/datasets/omni/omni_min_2016_withDate.csv')
    omni_data18_csv = read_csv(file = path+'MasterThesis/datasets/omni/omni_min_2018_withDate.csv')
    omni_data20_csv = read_csv(file = path+'MasterThesis/datasets/omni/omni_min_2020_withDate.csv')

# Dataset containing data
def files(original=True, green=True, train=False, mean=False):

    if train:
        folder = '/itf-fi-ml/home/koolsen/Aurora/Data/Aurora_train'
        json_file = '/itf-fi-ml/home/koolsen/Master/Full_aurora_NEW.json' # To large fil for GitHub
        csv_file = '/itf-fi-ml/home/koolsen/Master/MasterThesis/datasets/Full_aurora_NEW.csv'
        wl = '5577 and 6300'

    if green:
        wl = '5577'
        if original:
            folder = path+'T_DATA_green'
            json_file = path+'Aurora_G.json' # To large fil for GitHub
            csv_file = path+'MasterThesis/datasets/Aurora_G.csv'
            if mean:
                json_file_mean = path+'Aurora_G_omni_mean.json' # To large fil for GitHub
                csv_file_mean = path+'MasterThesis/datasets/Aurora_G_omni_mean.csv'
            else:
                json_file_omni = path+'Aurora_G_omni.json' # To large fil for GitHub
                csv_file_omni = path+'MasterThesis/datasets/Aurora_G_omni.csv'

        else:
            folder = path+'T_DATA_4yr_G'
            json_file = path+'Aurora_4yr_G.json' # To large fil for GitHub
            csv_file = path+'MasterThesis/datasets/Aurora_4yr_G.csv'
            if mean:
                json_file_mean = path+'Aurora_4yr_G_omni_mean.json' # To large fil for GitHub
                csv_file_mean = path+'MasterThesis/datasets/Aurora_4yr_G_omni_mean.csv'
            else:
                json_file_omni = path+'Aurora_4yr_G_omni.json' # To large fil for GitHub
                csv_file_omni = path+'MasterThesis/datasets/Aurora_4yr_G_omni.csv'

    else:
        wl = '6300'
        if original:
            folder = path+'T_DATA'
            json_file = path+'Aurora_R.json' # To large fil for GitHub
            csv_file = path+'MasterThesis/datasets/Aurora_R.csv'
            if mean:
                json_file_mean = path+'Aurora_R_omni_mean.json' # To large fil for GitHub
                csv_file_mean = path+'MasterThesis/datasets/Aurora_R_omni_mean.csv'
            else:
                json_file_omni = path+'Aurora_R_omni.json' # To large fil for GitHub
                csv_file_omni = path+'MasterThesis/datasets/Aurora_R_omni.csv'

        else:
            folder = path+'T_DATA_4yr_R'
            json_file = path+'Aurora_4yr_R.json' # To large fil for GitHub
            csv_file = path+'MasterThesis/datasets/Aurora_4yr_R.csv'
            if mean:
                json_file_mean = path+'Aurora_4yr_R_omni_mean.json' # To large fil for GitHub
                csv_file_mean = path+'MasterThesis/datasets/Aurora_4yr_R_omni_mean.csv'
            else:
                json_file_omni = path+'Aurora_4yr_R_omni.json' # To large fil for GitHub
                csv_file_omni = path+'MasterThesis/datasets/Aurora_4yr_R_omni.csv'


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


def average_omni_values(index, omni_data, omni_data_dates, tp, N_min):
    """
    Use solar wind data from 1 hour before timepoint.
    On night side
    Make 30 min (?) average Bz value (and the others)

    Some std thing?
    """

    ii = test_new(omni_data_dates, tp)
    index = ii

    solarwind = dict()
    indexes_min = []
    indexes_plus = []

    for i in range(1, N_min+1):

        indexes_min.append(index - i)
        indexes_plus.append(index + i)
        i += 1

    indexes_min.sort()
    index = [index]
    joinedlist = indexes_min + index + indexes_plus

    # Remove Negative Elements in List
    indexes = [ele for ele in joinedlist if ele > 0]

    for k in range(len(SW)):

        SW_value = []
        SW_remove = []
        for j in range(len(indexes)):

            index = [indexes[j]]
            #BZ_value = omni_data.loc[omni_data.index[index]][SW[0]].iloc[0]
            Value = omni_data.loc[omni_data.index[index]][SW[k]].iloc[0]
            #if BZ_value == 9999.99:
            if Value == 9999.99: # No data for Bz
                #print(Value)
                SW_remove.append(Value)
                #V_Bz = 9999.99
                continue
            elif Value == 99999.9: # No data for Speed
                #print(Value)
                SW_remove.append(Value)
                #V_speed = 99999.9
                continue
            elif Value == 999.99: # No data for Density
                #print(Value)
                SW_remove.append(Value)
                #V_density = 999.99
                continue
            else:
                SW_value.append(Value)

        #print(SW_value)
        #print(len(SW_value))
        #if len(str(SW_value)) == 1:
        if len(SW_value) == 0:

            Mean_BZ = np.mean(SW_remove, dtype = np.float64)
            Mean_BZ = "%.2f" % Mean_BZ
            SD_BZ = np.std(SW_remove)
            SD_BZ = "%.2f" % SD_BZ

            solarwind[SW[k]] = Mean_BZ
            solarwind[SW_SD[k]] = SD_BZ

        else:
            Mean_BZ = np.mean(SW_value, dtype = np.float64)
            Mean_BZ = "%.2f" % Mean_BZ
            SD_BZ = np.std(SW_value)
            SD_BZ = "%.2f" % SD_BZ

            solarwind[SW[k]] = Mean_BZ
            solarwind[SW_SD[k]] = SD_BZ

    return solarwind

def match_dates_omni_aurora_data(omni_data, omni_data_dates, tp, mean=True):

    index = 0
    dayside = ['06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17']

    #ii = test_new(omni_data_dates, tp)
    #index = ii
    #print(index)

    #for i in range(len(omni_data['Date'])):
    #    if tp in omni_data['Date'][i]:
    #        index.append(i)

    #print(index)
    #print(omni_data.loc[omni_data.index[index]]["Bz, nT (GSE)"])
    #print(omni_data.loc[omni_data.index[index]]["Bz, nT (GSM)"])
    #print(omni_data.loc[omni_data.index[ii]]["Bz, nT (GSM)"])
    #print(SW[0])

    if mean:

        hour = tp[-8:-6]
        if hour in dayside:
            # Dayside
            N_min = 3 # \pm 3 minutes
            solarwind = average_omni_values(index, omni_data, omni_data_dates, tp, N_min)

        else:
            # Nightside. 1 hour time diff.
            N_min = 15 # \pm 15 minutes
            tp_new = int(hour) - 1 # Get omni data for one hour before aurora image

            if len(str(tp_new)) == 1:
                tp_new =  '0{}'.format(tp_new)
                #print(tp_new)
            #if tp_new == 0:
            #    tp_new = '00'
            elif tp_new < 0:
                tp_new = hour
            tp_new = '{}{}{}'.format(tp[:-8], str(tp_new), tp[-6:])
            #print('tp: ', tp, 'tp new: ', tp_new)

            solarwind = average_omni_values(index, omni_data, omni_data_dates, tp_new, N_min)

    else:
        solarwind = dict()
        index = [index]
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

# Add information to data set
def add_file_information(json_file, csv_file):

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

    print("Json file updated with additional information")
    print(container)
    container.to_json(json_file)
    formats(json_file, csv_file)

# Add omni values to data set
def add_omni_information(json_file, json_file_, csv_file_, mean=True):

    container = DatasetContainer.from_json(json_file)
    print("length container: ", len(container))

    # Remove data for Feb, Mar, Oct
    counter = 0
    for i in range(len(container)):
        i -= counter
        if container[i].timepoint[5:7] == '02' \
        or container[i].timepoint[5:7] == '03' \
        or container[i].timepoint[5:7] == '10':
            del container[i]
            counter += 1
    print('removed images from container: ', counter)
    print('new container len: ', len(container))


    for entry in tqdm(container):

        # make solar wind data by matchind dates
        tp = entry.timepoint

        if tp[-2:] != "00":
            tp = tp[:-2] + "00"

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

        solarwind = match_dates_omni_aurora_data(omni_data, omni_data_dates, tp, mean)
        #print(solarwind)

        # Add solar wind data (dict) to json file
        entry.add_solarwind(solarwind)

    print("Json file updated with solarwind information")
    print(container)

    container.to_json(json_file_)
    formats(json_file_, csv_file_)


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

'''
start = time.time()

# green aurora, 2016 and 2018
folder, json_file, csv_file, wl = files(original=False)
json_file_mean = path+'Aurora_4yr_G_omni_mean.json' # To large fil for GitHub
csv_file_mean = path+'MasterThesis/datasets/Aurora_4yr_G_omni_mean.csv'
#file_from_ASIfolder(folder, wl, json_file)
#add_file_information(json_file, csv_file)
add_omni_information(json_file, json_file_mean, csv_file_mean)

stop = time.time() - start
print("Time [h] (set 1618): ", stop/(60*60))
'''
start = time.time()

# green aurora, 2014 and 2020
folder, json_file, csv_file, wl = files()
json_file_mean = path+'Aurora_G_omni_mean.json' # To large fil for GitHub
csv_file_mean = path+'MasterThesis/datasets/Aurora_G_omni_mean.csv'
#file_from_ASIfolder(folder, wl, json_file)
#add_file_information(json_file, csv_file)
add_omni_information(json_file, json_file_mean, csv_file_mean)

stop = time.time() - start
print("Time [h] (set 1420): ", stop/(60*60))
