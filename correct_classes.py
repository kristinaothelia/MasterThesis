from lbl.class_corrector import ClassCorrector
from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer
import numpy as np
"""
Description ...

Classes
0: aurora-less
1: arc
2: diffuse
3: discrete

Prediction level 0: checks human_prediction = False
Prediction level 1: checks human_prediction = None
Prediction level 2: checks human_prediction = True
"""

def class_correction(pred_level, json_from='', json_to='', arc=False, diff=False, disc=False, noa=False):
    if pred_level < 0 or pred_level > 2:
        print("Unvalid prediction level. Use 0 for false, 1 for none/null, 2 for true.]")
        pred_level = input("New prediction level: ")

    if arc == True:
        label = 'arc'
    elif diff == True:
        label = 'diffuse'
    elif disc == True:
        label = 'discrete'
    elif diff == True:
        label = 'aurora-less'
    else:
        print("No label selected")
    print(label)

    container = DatasetContainer.from_json(json_from)
    corrector = ClassCorrector(container=container)

    corrector.correct_class(label, prediction_level=pred_level, save_path=json_to)


#predicted_file = 'datasets/5577_k_aurora.json'
#corrected_file = 'datasets/5577_k_aurora_correct.json'
#class_correction(pred_level=0, json_from=predicted_file, json_to=corrected_file, arc=True, diff=False, disc=False, noa=False)



def correct_6300():
    container = DatasetContainer.from_json('6300_k.json')
    corrector = ClassCorrector(container=container)

    #corrector.correct_class(label='arc', prediction_level=1, save_path='6300_k.json')
    #corrector.correct_class(label='diffuse', prediction_level=1, save_path='6300_k.json')
    #corrector.correct_class(label='discrete', prediction_level=1, save_path='6300_k.json')
    corrector.correct_class(label='aurora-less', prediction_level=2, save_path='test_k.json')

def correct_5577():
    container = DatasetContainer.from_json('5577_k.json')
    corrector = ClassCorrector(container=container)

    #corrector.correct_class(label='arc', prediction_level=1, save_path='5577_k.json')
    #corrector.correct_class(label='diffuse', prediction_level=1, save_path='5577_k.json')
    #corrector.correct_class(label='discrete', prediction_level=1, save_path='5577_k.json')
    corrector.correct_class(label='aurora-less', prediction_level=1, save_path='5577_k.json')

# Correct 630.0nm, green aurora
#correct_6300()

# Correct 557.7nm, red aurora
#correct_5577()

'''
Eks: Lage en tabell med predicted vs observed (images needs to have a label)
OR
when correcting 'Full_aurora_predicted_local.json', make a table over correct/corrected

'''


# Correct 'Full_aurora_predicted' used with b0
#predicted_file = 'datasets/Full_aurora_predicted.json'  # json_from
predicted_file = 'datasets/Full_aurora_predicted_local.json'  # json_from
corrected_file = 'datasets/Full_aurora_predicted_b0_correct_arc.json' # json_to

container = DatasetContainer.from_json(predicted_file)

n_less = 0; n_arc = 0; n_diff = 0; n_disc = 0; tot = 0

for entry in container:

    if entry.human_prediction == False:
        tot += 1
        if entry.label == 'arc':
            n_arc += 1
        elif entry.label == 'diffuse':
            n_diff += 1
        elif entry.label == 'discrete':
            n_disc += 1
        else:
            n_less += 1

print('tot:        ', tot, 'added: ', (n_arc + n_less + n_diff + n_disc))
print("Aurora-less ", n_less)
print("Arc         ", n_arc)
print("Diffuse     ", n_diff)
print("Discrete    ", n_disc)


corrector = ClassCorrector(container=container)

#corrector.correct_class('arc', prediction_level=0, save_path=corrected_file)
corrector.correct_class('diffuse', prediction_level=0, save_path=corrected_file)
