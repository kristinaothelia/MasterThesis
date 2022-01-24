from lbl.class_corrector import ClassCorrector
from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer
import numpy as np
import matplotlib.pyplot as plt
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

stats(container, pred_level=False)
stats(container, pred_level=True)

corrector = ClassCorrector(container=container)
corrector.correct_class(LABELS[0], prediction_level=0, save_path=corrected_file)

stats(container, pred_level=False)
stats(container, pred_level=True)
