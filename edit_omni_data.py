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
    omni_data14_csv = read_csv(file = '/itf-fi-ml/home/koolsen/Master/MasterThesis/datasets/omni/omni_min_2014_withDate.csv')
    omni_data16_csv = read_csv(file = '/itf-fi-ml/home/koolsen/Master/MasterThesis/datasets/omni/omni_min_2016_withDate.csv')
    omni_data18_csv = read_csv(file = '/itf-fi-ml/home/koolsen/Master/MasterThesis/datasets/omni/omni_min_2018_withDate.csv')
    omni_data20_csv = read_csv(file = '/itf-fi-ml/home/koolsen/Master/MasterThesis/datasets/omni/omni_min_2020_withDate.csv')

'''
@jit()  # nopython=True
def test_new(omni, timepoint):
    for i in range(len(omni)):
        if timepoint in omni[i]:
            return i
'''
def match_dates_omni_aurora_data(omni_data, omni_data_dates, tp, SW):

    index = 0
    #index = []
    solarwind = dict()

    dayside = ['06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17']
    #dayside = ['06', '07', '08', '09', 10, 11, 12, 13, 14, 15, 16, 17]

    def NAVN(omni_data, omni_data_dates, tp, SW, N_min):
        '''
        ii = test_new(omni_data_dates, tp)
        index.append(ii)
        #print(index)
        '''
        print(tp)

        for i in range(len(omni_data['Date'])):
            if tp in omni_data['Date'][i]:
                #index.append(i)
                index = i

        indexes_min = []
        indexes_plus = []

        for i in range(1, N_min+1):

            indexes_min.append(index - i)
            indexes_plus.append(index + i)
            i += 1

        indexes_min.sort()

        #print(index)
        #print(omni_data.loc[omni_data.index[index]]["Bz, nT (GSE)"])
        #print(omni_data.loc[omni_data.index[index]]["Bz, nT (GSM)"])
        #print(omni_data.loc[omni_data.index[ii]]["Bz, nT (GSM)"])
        #print(SW[0])

        index = [index]

        joinedlist = indexes_min + index + indexes_plus
        #print(joinedlist)

        # Remove Negative Elements in List
        indexes = [ele for ele in joinedlist if ele > 0]
        print(indexes)
        #SW_value = []

        for k in range(len(SW)):
            SW_value = []
            #print(SW[k])
            #print(omni_data.loc[omni_data.index[index]][SW[k]].iloc[0])
            #if k == 0:
            for j in range(len(indexes)):

                index = [indexes[j]]
                #BZ_value = omni_data.loc[omni_data.index[index]][SW[0]].iloc[0]
                Value = omni_data.loc[omni_data.index[index]][SW[k]].iloc[0]
                #if BZ_value == 9999.99:
                if Value == 9999.99: # No data for Bz
                    print(Value)
                    continue
                elif Value == 99999.9: # No data for Speed
                    print(Value)
                    continue
                elif Value == 999.99: # No data for Density
                    print(Value)
                    continue
                else:
                    SW_value.append(Value)

            print(SW_value)
            print(len(SW_value))
            Mean_BZ = np.mean(SW_value, dtype = np.float64)
            Mean_BZ = "%.2f" % Mean_BZ
            SD_BZ = np.std(SW_value)
            SD_BZ = "%.2f" % SD_BZ

            solarwind[SW[k]] = Mean_BZ
            solarwind[SW_SD[k]] = SD_BZ

            '''
            else:
                solarwind[SW[k]] = omni_data.loc[omni_data.index[index]][SW[k]].iloc[0]
                solarwind[SW_SD[k]] = 10
            '''


    hour = tp[-8:-6]
    if hour in dayside:
        # Dayside
        N_min = 3 # \pm 3 minutes
        NAVN(omni_data, omni_data_dates, tp, SW, N_min)

    else:
        # Nightside. 1 hour time diff.
        N_min = 15 # \pm 15 minutes
        #if tp[:-6] == '2014-12-31 00':
        #    tp_new[:-6] = '2016-12-30 00'

        tp_new = int(hour) - 1 # Get omni data for one hour before aurora image

        if len(str(tp_new)) == 1:
            tp_new =  '0{}'.format(tp_new)
            print(tp_new)
        #if tp_new == 0:
        #    tp_new = '00'
        elif tp_new < 0:
            tp_new = hour
        tp_new = '{}{}{}'.format(tp[:-8], str(tp_new), tp[-6:])

        NAVN(omni_data, omni_data_dates, tp_new, SW, N_min)


    #Bz_GSE = omni_data.loc[omni_data.index[index]]["Bz, nT (GSE)"]
    #Bz_GSM = omni_data.loc[omni_data.index[index]]["Bz, nT (GSM)"]
    #values.append(Bz_GSE.iloc[0])
    #values.append(Bz_GSM.iloc[0])

    return solarwind

# Dayside
img_time = '2020-01-01 08:05:00'

# Nightside
img_time = '2020-12-31 23:55:00'

img_time = '2016-12-31 23:53:00'

new_Bz = []

def test(tp, SW):

    print(tp)

    if tp[:4] == "2014":
        omni_data = omni_data14_csv

    if tp[:4] == "2016":
        omni_data = omni_data16_csv

    if tp[:4] == "2018":
        omni_data = omni_data18_csv

    if tp[:4] == "2020":
        omni_data = omni_data20_csv

    #print(omni_data)

    # get only the dates
    omni_data_dates = omni_data['Date']
    omni_data_dates = omni_data_dates.values

    solarwind = match_dates_omni_aurora_data(omni_data, omni_data_dates, tp, SW)

    print(solarwind)

test(img_time, SW)
