from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from pylab import *
# -----------------------------------------------------------------------------

def remove(container):
    # Remove data for Feb, Mar, Oct in a container
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

    container.to_json(r'C:\Users\Krist\Documents\ASI_json_files\Aurora_G_omni_mean_predicted_efficientnet-b2_cut.json')


# All 4 years, jan+nov+dec
predicted_G_Full = r'C:\Users\Krist\Documents\ASI_json_files\AuroraFull_G_omni_mean_predicted_efficientnet-b2.json'

def split(container, nightside=False):

    day_start = 6
    day_end = 17

    if nightside:
        print('nightside')
        counter = 0

        for i in range(len(container)):

            i -= counter

            if int(container[i].timepoint[-8:-6]) >= day_start and int(container[i].timepoint[-8:-6]) <= day_end:

                #print(container[i].timepoint[-8:-6])
                del container[i]
                counter += 1
        print('removed images from container: ', counter)
        print('new container len: ', len(container))

        container.to_json(r'C:\Users\Krist\Documents\ASI_json_files\AuroraFull_G_omni_mean_predicted_efficientnet-b2_nighttime.json')
        return container

    else:
        print('dayside')
        counter = 0

        for i in range(len(container)):

            i -= counter

            if int(container[i].timepoint[-8:-6]) >= day_start and int(container[i].timepoint[-8:-6]) <= day_end:
                #counter += 1
                continue

            else:
                #print(container[i].timepoint[-8:-6])
                del container[i]
                counter += 1

        #print('removed images from container: ', counter)
        print('new container len: ', len(container))

        container.to_json(r'C:\Users\Krist\Documents\ASI_json_files\AuroraFull_G_omni_mean_predicted_efficientnet-b2_daytime.json')
        return container

def split_container(container, nightside=False):

    container = split(container, nightside)

    times = []
    for entry in container:
        if entry.timepoint[-8:-6] not in times:
            times.append(entry.timepoint[-8:-6])
    print(times)

    return container

container_Full = DatasetContainer.from_json(predicted_G_Full)
container_D = split_container(container_Full)

container_Full = DatasetContainer.from_json(predicted_G_Full)
container_N = split_container(container_Full, True)
