from lbl.class_corrector import ClassCorrector
from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
from tqdm import tqdm
import torch
# -----------------------------------------------------------------------------
"""
Description ...

Prediction level 0: checks human_prediction = False
Prediction level 1: checks human_prediction = None
Prediction level 2: checks human_prediction = True

Use False if labels are added by model, for runthrough of predicted labels
Use None if image has no label
Use True if image has label that is labeled by a human, for check
"""

LABELS = ['aurora-less', 'arc', 'diffuse', 'discrete']

#predicted_file = 'datasets/Full_aurora_new_rt_predicted_efficientnet-b3.json'
predicted_file = 'datasets/Full_aurora_new_rt_predicted_efficientnet-b3_TESTNEW_.json'
corrected_file = 'datasets/Full_aurora_new_rt_predicted_efficientnet-b3_TESTNEW_.json'

container = DatasetContainer.from_json(predicted_file)
length_container = len(container)
print("length of dataset/container: ", length_container)

def makeTest_json():

    folder = r'C:\Users\Krist\Documents\dataset_subfolders\arc_test'
    folder = r'C:\Users\Krist\Documents\dataset_subfolders\Test'
    wl = '5577 and 6300'
    container = DatasetContainer()
    container.from_folder(path=folder,
                          datasetname='green',
                          dataset_type='png',
                          wavelength=wl,
                          source='UiO',
                          location='Svaalbard',
                          dataset_description='ASI')

    for entry in container:
        # Add wavelength to json file
        entry.wavelength = container[0]['image_path'][-12:-8]

        path = Path(entry.image_path).stem
        date = path.split('_')[1]
        timestamp = path.split('_')[2]
        date = datetime.date(year=int(date[:4]), month=int(date[4:6]), day=int(date[6:]))
        timestamp = datetime.time(hour=int(timestamp[:2]), minute=int(timestamp[2:4]), second=int(timestamp[4:]))

        tiime = datetime.datetime(year=date.year, month=date.month, day=date.day, hour=timestamp.hour, minute=timestamp.minute, second=timestamp.second)

        entry.timepoint = str(tiime)

    # Labled data
    arcs        = Path(r'C:\Users\Krist\Documents\dataset_subfolders\arc_test')
    noaurora    = Path(r'C:\Users\Krist\Documents\dataset_subfolders\no_aurora_test')
    diffuse     = Path(r'C:\Users\Krist\Documents\dataset_subfolders\diffuse_test')
    discrete    = Path(r'C:\Users\Krist\Documents\dataset_subfolders\discrete_test')

    arcs = arcs.absolute()
    arc_files = list(arcs.glob('*'))

    diffuse = diffuse.absolute()
    diffuse_files = list(diffuse.glob('*'))

    discrete = discrete.absolute()
    discrete_files = list(discrete.glob('*'))

    noaurora = noaurora.absolute()
    noaurora_files = list(noaurora.glob('*'))

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

    container.to_json('datasets/Full_aurora_test_set.json')
    #container.to_json('datasets/Full_aurora_test_set.json')
    container = DatasetContainer.from_json('datasets/Full_aurora_test_set.json')
    print(len(container))

#makeTest_json()

def New_dataset(container, containerTest):
    counter = 0
    '''
    length = len(container)
    print('original container length:   ', length)
    counter = 0

    for i in range(length):
        i -= counter
        if container[i].label == None:
            del container[i]
            counter += 1
    '''

    length = len(container)
    print('original container length:   ', length)
    for i in range(length):
        i -= counter

        for entryT in containerTest:
            if Path(container[i].image_path).stem == Path(entryT.image_path).stem:
                counter += 1
                del container[i]

    print('removed images: ', counter)
    return container

json_file = 'datasets/Full_aurora_new_rt_ml_predicted_efficientnet-b3_TESTNEW_.json'
container = DatasetContainer.from_json(json_file)
containerTest = DatasetContainer.from_json('datasets/Full_aurora_test_set.json')

#new_container = New_dataset(container, containerTest)
#new_container.to_json('datasets/Full_aurora_train_valid_set.json')

train_container = DatasetContainer.from_json('datasets/Full_aurora_train_valid_set.json')
test_container = DatasetContainer.from_json('datasets/Full_aurora_test_set.json')
print(len(train_container))
print(len(test_container))
print(len(test_container)+len(train_container))
exit()



def test(container):
    counter = 0

    for entry in container:

        def keywithmaxval(d):
            v=list(d.values())
            k=list(d.keys())
            return k[v.index(max(v))]
        pred = keywithmaxval(entry.score)

        if entry.label != pred:

            #if entry.label == LABELS[0]:
            #    counter += 1

            #print(entry.label, pred)
            #print("Filename: ", Path(entry.image_path).stem)
            #img = entry.open()
            #plt.title('Label: {}. Pred: {}'.format(entry.label, pred))
            #plt.imshow(img) #, cmap='gray'
            #plt.show()
            counter += 1
    print("nr. of entries with different label and pred: ", counter)

test(container)
print("total number of entries: ", len(container))
exit()

def stats(container, pred_level=False):

    n_less = 0; n_arc = 0; n_diff = 0; n_disc = 0; tot = 0

    for entry in container:

        if entry.human_prediction == pred_level:  # False, True
            tot += 1

            if entry.label == LABELS[1]:
                n_arc += 1
            elif entry.label == LABELS[2]:
                n_diff += 1
            elif entry.label == LABELS[3]:
                n_disc += 1
            elif entry.label == LABELS[0]:
                n_less += 1

    if tot == 0:
        print("No images with this prediction level")
    else:
        print("\npred_level: ", pred_level)
        print("%23s: %g (%3.1f%% of dataset)" %('Total classified images', tot, (tot/length_container)*100))
        print("%23s: %4g (%3.1f%%)" %(LABELS[0], n_less, (n_less/tot)*100))
        print("%23s: %4g (%3.1f%%)" %(LABELS[1], n_arc, (n_arc/tot)*100))
        print("%23s: %4g (%3.1f%%)" %(LABELS[2], n_diff, (n_diff/tot)*100))
        print("%23s: %4g (%3.1f%%)" %(LABELS[3], n_disc, (n_disc/tot)*100))

#stats(container, pred_level=False)
#stats(container, pred_level=True)

#corrector = ClassCorrector(container=container)

#corrector.correct_class(LABELS[1], prediction_level=0, save_path=corrected_file)


def two_files(container1, container2):

    counter = 0

    for entry in container1:
        im1 = Path(entry.image_path).stem

        for entry2 in container2:

            if im1 == Path(entry2.image_path).stem:

                if entry.label != entry2.label:
                    counter += 1

                #print(entry.label)
                #print(entry2.label)

        #exit()
    print(counter)

def two_files_noA(container1, container2):

    counter = 0
    list = []

    for entry in container1:
        im1 = Path(entry.image_path).stem

        for entry2 in container2:

            if im1 == Path(entry2.image_path).stem:

                if entry.label == LABELS[0] or entry2.label == LABELS[0]:

                    if entry.label != entry2.label:
                        counter += 1
                        list.append(str(im1))

                        #print("{}:{} and {}".format(im1, entry.label, entry2.label))

                    #print(entry.label)
                    #print(entry2.label)

        #exit()
    print(counter)
    print(len(list))
    return list

container1 = container
container2 = DatasetContainer.from_json('datasets/Full_aurora_ml_corr_NEW.json')
print("total number of entries (set 2): ", len(container2))
#two_files(container1, container2)

def remove_img(container, list):
    """ Remove images with no label """

    length = len(container)
    print('original container length:   ', length)
    counter = 0
    for i in range(length):
        i -= counter
        if Path(container[i].image_path).stem in list:
            del container[i]
            counter += 1
    return container

'''
list = two_files_noA(container1, container2)
c = DatasetContainer.from_json('datasets/Full_aurora_ml_corr_NEW.json')
container_new = remove_img(c, list)
container_new.to_json('container_new.json')
stats(container_new, pred_level=False)
stats(container_new, pred_level=True)
exit()
'''
container_new = DatasetContainer.from_json('container_new.json')
corrected_file = 'Full_aurora_new_rt_predicted_efficientnet-b3_TESTNEW_NEW.json'
# Update list entries
f = DatasetContainer.from_json(corrected_file)
print(len(f))
stats(f, pred_level=False)
stats(f, pred_level=True)
exit()
stats(container_new, pred_level=False)
stats(container_new, pred_level=True)
corrector = ClassCorrector(container=container_new)
corrector.correct_class(LABELS[2], prediction_level=2, save_path=corrected_file)
