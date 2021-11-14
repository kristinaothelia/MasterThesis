from pathlib import Path
import datetime
from tqdm import tqdm
import sys

import matplotlib.pyplot as plt

from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer

# Dataset containing all data
#folder = '/home/jon/Documents/LBL/all/dataset/6300'
folder = r'C:\Users\Krist\Documents\dataset\5577'

container = DatasetContainer()
container.from_folder(path=folder,
                      datasetname='green',
                      dataset_type='png',
                      source='UiO',
                      location='Svaalbard',
                      dataset_description='ASI')

# container.to_json('files.json')
# container = DatasetContainer.from_json('files.json')


for entry in container:
    path = Path(entry.image_path).stem
    date = path.split('_')[1]
    timestamp = path.split('_')[2]
    date = datetime.date(year=int(date[:4]), month=int(date[4:6]), day=int(date[6:]))
    timestamp = datetime.time(hour=int(timestamp[:2]), minute=int(timestamp[2:4]), second=int(timestamp[4:]))

    tiime = datetime.datetime(year=date.year, month=date.month, day=date.day, hour=timestamp.hour, minute=timestamp.minute, second=timestamp.second)

    entry.timepoint = str(tiime)

'''
arcs = Path('/home/jon/Documents/LBL/6300_original/Data_5classes/Arc')
clear = Path('/home/jon/Documents/LBL/6300_original/Data_5classes/Clear')
cloud = Path('/home/jon/Documents/LBL/6300_original/Data_5classes/Cloud')
diffuse = Path('/home/jon/Documents/LBL/6300_original/Data_5classes/Diffuse')
discrete = Path('/home/jon/Documents/LBL/6300_original/Data_5classes/Discrete')
'''
# Labled data
arcs        = Path(r'C:\Users\Krist\Documents\5577_add\Transf_jpeg\Data_4classes\Arc')
noaurora    = Path(r'C:\Users\Krist\Documents\5577_add\Transf_jpeg\Data_4classes\No_aurora')
diffuse     = Path(r'C:\Users\Krist\Documents\5577_add\Transf_jpeg\Data_4classes\Diffuse')
discrete    = Path(r'C:\Users\Krist\Documents\5577_add\Transf_jpeg\Data_4classes\Discrete')

arcs = arcs.absolute()
arc_files = list(arcs.glob('*'))

diffuse = diffuse.absolute()
diffuse_files = list(diffuse.glob('*'))

discrete = discrete.absolute()
discrete_files = list(discrete.glob('*'))

noaurora = noaurora.absolute()
noaurora_files = list(noaurora.glob('*'))

'''
clear = clear.absolute()
clear_files = list(clear.glob('*'))

cloud = cloud.absolute()
cloud_files = list(cloud.glob('*'))

cloud_files.extend(clear_files)
'''


for entry in tqdm(container):

    for file in arc_files:
        file = file.stem
        if Path(entry.image_path).stem == file:
            entry.label = 'arc'
            break

    for file in diffuse_files:
        file = file.stem
        if Path(entry.image_path).stem == file:
            entry.label = 'diffuse'
            break

    for file in discrete_files:
        file = file.stem
        if Path(entry.image_path).stem == file:
            entry.label = 'discrete'
            break

    for file in noaurora_files:
        file = file.stem
        if Path(entry.image_path).stem == file:
            entry.label = 'aurora-less'
            break

container.to_json('datasets/5577_k_aurora.json')
