from pathlib import Path
import datetime
from tqdm import tqdm
import sys

import matplotlib.pyplot as plt

from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer

# Dataset containing data
folder = r'C:\Users\Krist\Documents\MasterThesis\JSON_TEST'

container = DatasetContainer()
container.from_folder(path=folder,
                      datasetname='New data for testing on cnn model',
                      dataset_type='png',
                      source='UiO',
                      location='Svaalbard',
                      dataset_description='ASI')

container.to_json('files_new.json')
container = DatasetContainer.from_json('files_new.json')

for entry in container:
    path = Path(entry.image_path).stem
    entry.wavelength = container[0]['image_path'][-12:-8]
    date = path.split('_')[1]
    timestamp = path.split('_')[2]
    date = datetime.date(year=int(date[:4]), month=int(date[4:6]), day=int(date[6:]))
    timestamp = datetime.time(hour=int(timestamp[:2]), minute=int(timestamp[2:4]), second=int(timestamp[4:]))

    tiime = datetime.datetime(year=date.year, month=date.month, day=date.day, hour=timestamp.hour, minute=timestamp.hour, second=timestamp.second)

    entry.timepoint = str(tiime)

container.to_json('files_new.json')
