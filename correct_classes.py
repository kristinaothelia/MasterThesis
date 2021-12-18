from lbl.class_corrector import ClassCorrector
from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer
import numpy as np
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------
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
LABELS = ['aurora-less', 'arc', 'diffuse', 'discrete']
'''
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
'''

#predicted_file = 'datasets/Full_aurora_predicted_b3.json'
predicted_file = 'datasets/Full_aurora_new_rt.json'

#corrected_file = 'datasets/Full_aurora_predicted_b0_correct_arc.json' # json_to
corrected_file = 'datasets/Full_aurora_new_rt.json' # json_to

container = DatasetContainer.from_json(predicted_file)
print(len(container))
''' eks:
"timepoint": "2015-11-17 18:18:37",
"label": null,
"human_prediction": null,
"score": {}
'''

n_less = 0; n_arc = 0; n_diff = 0; n_disc = 0; tot = 0

for entry in container:

    if entry.human_prediction == True:  # None
        tot += 1
        if entry.label == LABELS[1]:
            n_arc += 1
        elif entry.label == LABELS[2]:
            n_diff += 1
        elif entry.label == LABELS[3]:
            n_disc += 1
        elif entry.label == LABELS[0]:
            n_less += 1
        #else:
        #    n_less += 1

print("%23s: %g" %('Total classified images', tot))
print("%23s: %4g (%3.1f%%)" %(LABELS[0], n_less, (n_less/tot)*100))
print("%23s: %4g (%3.1f%%)" %(LABELS[1], n_arc, (n_arc/tot)*100))
print("%23s: %4g (%3.1f%%)" %(LABELS[2], n_diff, (n_diff/tot)*100))
print("%23s: %4g (%3.1f%%)" %(LABELS[3], n_disc, (n_disc/tot)*100))

exit()

corrector = ClassCorrector(container=container)

label = LABELS[0]
corrector.correct_class(label, prediction_level=2, save_path=corrected_file)
