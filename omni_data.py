import numpy as np
import pandas as pd
import json
import os
import time
import datetime
import math

from numba import cuda, jit
#print(cuda.gpus)

from tqdm import tqdm
from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer
'''
#json_file = 'datasets/Full_aurora_predicted_b2.json'
#json_file = 'datasets/t_data_with_2014nya4.json'
json_file = '/itf-fi-ml/home/koolsen/Master/t_data_green_with_2014nya4.json'
#json_file = 't_data_with_2014nya4_predicted_b2.json'
xlsx_file = '/itf-fi-ml/home/koolsen/Master/t_data_green_with_2014nya4.xlsx'
#xlsx_file = 'datasets/json_to_xls.xlsx'
csv_file  = '/itf-fi-ml/home/koolsen/Master/t_data_green_with_2014nya4.csv'
#csv_file  = 'datasets/xls_to_csv.csv'
file14 = "..\omni\omni_min2014.xlsx"
file20 = "..\omni\omni_min2020.xlsx"
'''
# Excel file to csv file:
def read_excel(file):
    start = time.time()
    data = pd.read_excel(file)
    end = time.time()
    print("Time to read excel (%s): %.2f min" %(file, (end-start)/60))

    return data

def xls_to_csv(data, name):
    data.to_csv(name, index=False)
    print("From csv file")
    print(data)

def read_csv(file, print=False):

    data = pd.read_csv(file)
    if print:
        print("From csv file (%s)" %file)
        #data = pd.read_csv(file, index_col=[0])
        print(data)

    return data

# json file to excel file
def json_to_xls(json_file):
    with open(json_file) as json_file:
        data = json.load(json_file)

    df = pd.DataFrame.from_dict(data['entries'])
    print(df['score'])

    df.to_excel(xlsx_file)
    print("json saved as excel file")
    xls_to_csv(df, csv_file)
    print("json saved as csv file")


'''
if os.path.isfile(xlsx_file) and os.path.isfile(csv_file):
    print("json file exists as excel and csv file")
else:
    print("json file do not exists as excel and/or csv file")
    json_to_xls(json_file)
'''
def edit_omni_dates_to_correct_form(omni_data):

    Year = omni_data["Year"]
    Day  = omni_data["Day"]
    omni_new = omni_data.drop(["Year", "Hour", "Minute"], axis=1)

    MONTH = []; DAY = []
    for i in range(len(Year)):

        month_day = datetime.datetime.strptime('{} {}'.format(Day[i], Year[i]),'%j %Y')
        MONTH.append(month_day.month)
        DAY.append(month_day.day)
        #print(month_day.month, month_day.day)

    df = pd.DataFrame()
    df['year'] = Year
    df['month'] = MONTH
    df['day'] = DAY
    df['hour'] = omni_data["Hour"]
    df['minute'] = omni_data["Minute"]

    new_df = pd.DataFrame()
    new_df["Date"] = pd.to_datetime(df)
    #print(new_df)
    omni_new['Day'] = new_df['Date'].values
    omni_new.rename(columns={'Day': 'Date'}, inplace=True)
    print(omni_new)

    return omni_new

def omni_to_csv(file, out_file):

    omni_data = read_excel(file=file)
    print(omni_data[:3])
    xls_to_csv(data=omni_data, name=out_file)


def correct_omni_data(file, out_file):
    # Make dataframe with final date values
    omni_data = read_csv(file=file)
    omni_new = edit_omni_dates_to_correct_form(omni_data)
    omni_new.to_csv(out_file, index=False)

omni_to_csv(file='datasets/omni/excel/omni_min_2014.xlsx', out_file='datasets\omni\omni_min_2014.csv')
correct_omni_data(file='datasets\omni\omni_min_2014.csv', out_file='datasets\omni\omni_min_2014_withDate.csv')

omni_to_csv(file='datasets/omni/excel/omni_min_2016.xlsx', out_file='datasets\omni\omni_min_2016.csv')
correct_omni_data(file='datasets\omni\omni_min_2016.csv', out_file='datasets\omni\omni_min_2016_withDate.csv')

omni_to_csv(file='datasets/omni/excel/omni_min_2018.xlsx', out_file='datasets\omni\omni_min_2018.csv')
correct_omni_data(file='datasets\omni\omni_min_2018.csv', out_file='datasets\omni\omni_min_2018_withDate.csv')

omni_to_csv(file='datasets/omni/excel/omni_min_2020.xlsx', out_file='datasets\omni\omni_min_2020.csv')
correct_omni_data(file='datasets\omni\omni_min_2020.csv', out_file='datasets\omni\omni_min_2020_withDate.csv')

exit()

'''
omni_data14 = read_csv(file='\omni\omni_min2014_withDate.csv')
#print(omni_data14.loc[:, 'Date'])
#print(len(omni_data14['Date']))

time = '2014-01-02'
time = '2014-12-31 23:59:37'
#time = time[:-3]    # remove seconds
time = time[:-6]

#if omni_data14['Date'] == '2014-01-02 04:05:00':
#    print("exists")
count = 0
index = []
for i in range(len(omni_data14['Date'])):
    #print(omni_data14['Date'][i])
    if time in omni_data14['Date'][i]:
        #print('TRUE')
        #print(omni_data14['Date'][i])
        index.append(i)
        #print(omni_data14.iloc[[i]])    # index
        count += 1

print(count)
print(index)
print(omni_data14.loc[omni_data14.index[index]])
#print(omni_data14.iloc[[index]])

#omni_data20 = read_csv(file='..\omni\omni_min2020_withDate.csv')
'''



# Test:

# csv dataframe to json file..?
# https://medium.com/@hannah15198/convert-csv-to-json-with-python-b8899c722f6d
# https://stackoverflow.com/questions/56113592/convert-csv-to-json-file-in-python
# https://pythonexamples.org/python-csv-to-json/


# DataFrames:
aurora_csv_file = read_csv("datasets/Aurora_R.csv") # red aurora data, 2014 and 2020, not predicted
#aurora_csv_file = read_csv("datasets/Aurora_G.csv") # green aurora data, 2014 and 2020, not predicted
omni14 = 'datasets\omni\omni_min_2014_withDate.csv'
omni20 = 'datasets\omni\omni_min_2020_withDate.csv'

if os.path.isfile(omni14):
    omni_data14_csv = read_csv(file=omni14)
    omni_data20_csv = read_csv(file=omni20)
else:
    omni_data14_csv = read_csv(file = '/itf-fi-ml/home/koolsen/Master/MasterThesis/datasets/omni/omni_min_2014_withDate.csv')
    omni_data20_csv = read_csv(file = '/itf-fi-ml/home/koolsen/Master/MasterThesis/datasets/omni/omni_min_2020_withDate.csv')

# function optimized to run on gpu
@jit()
def test_new(omni, timepoint):
    for i in range(len(omni)):
        if timepoint in omni[i]:
            return i

def match_dates_omni_aurora_data(omni_data, omni_data_dates, aurora_data, timepoint):

    '''
    for i in range(len(aurora_data)):
        #print(aurora_data["timepoint"][i])
        if time in aurora_data["timepoint"][i]:
            cropped_time = aurora_data["timepoint"][i]
            cropped_time = cropped_time[:-6]
            print("True ! ", aurora_data["timepoint"][i], " Cropped time: ", cropped_time)
    '''

    index = []
    ii = test_new(omni_data_dates, timepoint)
    index.append(ii)
    '''
    for i in range(len(omni_data['Date'])):
        #print(omni_data['Date'][i])
        if time in omni_data['Date'][i]:
            #print('TRUE')
            #print(omni_data['Date'][i])
            index.append(i)
            #print(omni_data.iloc[[i]])    # index
            #count += 1
    '''

    #print(index)
    #print(omni_data.iloc[[index]])
    #print(omni_data.loc[omni_data.index[index]])
    #print(omni_data.loc[omni_data.index[index]]["Bz, nT (GSE)"])
    #print(omni_data.loc[omni_data.index[index]]["Bz, nT (GSM)"])

    Bz_GSE = omni_data.loc[omni_data.index[index]]["Bz, nT (GSE)"]
    Bz_GSM = omni_data.loc[omni_data.index[index]]["Bz, nT (GSM)"]

    return Bz_GSE.iloc[0], Bz_GSM.iloc[0]


# New DataFrame to aurora and omni data
df_TEST = aurora_csv_file#.iloc[:3]
#print(df_TEST); print(df_TEST['timepoint'])

Bz_GSE_list = []
Bz_GSM_list = []
#start = time.time()
for i in tqdm(range(len(df_TEST))):

    timepoint = df_TEST["timepoint"][i]
    if timepoint[-2:] != "00":
        timepoint = timepoint[:-2] + "00"
        #print("30 sec mark, need editing. New time: ", time)

    if timepoint[:4] == "2014":
        omni_data = omni_data14_csv
    elif timepoint[:4] == "2020":
        omni_data = omni_data20_csv
    else:
        print("Wrong year input in aurora data")
        #exit()

    omni_data_dates = omni_data['Date']
    omni_data_dates = omni_data_dates.values

    #threadsperblock = len(omni_data)
    #blockspergrid = math.ceil(omni_data.shape[0] / threadsperblock)
    Bz_GSE, Bz_GSM = match_dates_omni_aurora_data(omni_data, omni_data_dates, aurora_csv_file, timepoint)
    #Bz_GSE, Bz_GSM = match_dates_omni_aurora_data[blockspergrid, threadsperblock](omni_data, aurora_csv_file, time)
    Bz_GSE_list.append(Bz_GSE)
    Bz_GSM_list.append(Bz_GSM)


#stop = time.time()
#print("Time: %.2f min" %(end-start)/60)

n = len(df_TEST.columns)

df_TEST.insert(n+1, 'Bz, nT (GSE)', Bz_GSE_list)
df_TEST.insert(n+2,'Bz, nT (GSM)', Bz_GSM_list)

print(df_TEST)

#df_TEST.to_csv("Aurora_G_Bz.csv", index=False)
df_TEST.to_csv("Aurora_R_Bz.csv", index=False)
#df_TEST.to_excel("Aurora_R_Bz.xls", index=False)
print("saved files with Bz values")


json_file = r'C:\Users\Krist\Documents\ASI_json_files\Aurora_R.json'
aurora_file = DatasetContainer.from_json(json_file)
