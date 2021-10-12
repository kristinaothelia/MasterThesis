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

# Correct 'Full_aurora_predicted' used with b0
#predicted_file = 'datasets/Full_aurora_predicted.json'  # json_from
#predicted_file = 'datasets/Full_aurora_predicted_local.json'  # json_from
#predicted_file = 'datasets/Full_aurora_predicted_b3.json'
predicted_file = 'datasets/Full_aurora_new_runthrough.json'
#corrected_file = 'datasets/Full_aurora_predicted_b0_correct_arc.json' # json_to
corrected_file = 'datasets/Full_aurora_new_runthrough.json' # json_to

container = DatasetContainer.from_json(predicted_file)

''' eks:
"timepoint": "2015-11-17 18:18:37",
"label": null,
"human_prediction": null,
"score": {}
'''


def stats(label='arc'):

    def autolabel(n):
        """
        Attach a text label above each bar displaying its height
        autolabel() from: https://stackoverflow.com/a/42498711
        """
        for i in n:
            height = i.get_height()
            ax.text(i.get_x() + i.get_width()/2., 1.01*height,
                    '%d' % int(height), ha='center', va='bottom')

    Month = []
    Years = []

    for entry in container:

        #if entry.human_prediction == True:
        if entry.label == label:

            Month.append(entry.timepoint[5:7])  # month
            Years.append(entry.timepoint[:4])

    M_label = ['Jan', 'Feb', 'Mar', 'Oct', 'Nov', 'Dec']
    M_ = ['01', '02', '03', '10', '11', '12']
    M = [Month.count(M_[0]), Month.count(M_[1]), Month.count(M_[2]),
         Month.count(M_[3]), Month.count(M_[4]), Month.count(M_[5])]

    fig, ax = plt.subplots()
    n = ax.bar(M_label, M)
    autolabel(n)
    plt.title("%s: for Jan, Feb, Mar, Oct, Nov, Dec (2014-2019)" %label)
    #plt.xticks(rotation='vertical')
    plt.xlabel("Month"); plt.ylabel("Count")


    Y_ = ['2014', '2015', '2016', '2017', '2018', '2019']
    Y = [Years.count(Y_[0]), Years.count(Y_[1]), Years.count(Y_[2]),
         Years.count(Y_[3]), Years.count(Y_[4]), Years.count(Y_[5])]

    fig, ax = plt.subplots()
    ny = ax.bar(Y_, Y)
    autolabel(ny)
    plt.title("%s: 2014-2019" %label)
    #plt.xticks(rotation=70)
    plt.xlabel("Year"); plt.ylabel("Count")


#stats(label='arc')
#stats(label='discrete')
#plt.show()


n_less = 0; n_arc = 0; n_diff = 0; n_disc = 0; tot = 0

for entry in container:

    if entry.human_prediction == True:  # False
        tot += 1
        if entry.label == LABELS[1]:
            n_arc += 1
        elif entry.label == LABELS[2]:
            n_diff += 1
        elif entry.label == LABELS[3]:
            n_disc += 1
        else:
            n_less += 1

print("%23s: %g" %('Total classified images', tot))
print("%23s: %4g (%3.1f%%)" %(LABELS[0], n_less, (n_less/tot)*100))
print("%23s: %4g (%3.1f%%)" %(LABELS[1], n_arc, (n_arc/tot)*100))
print("%23s: %4g (%3.1f%%)" %(LABELS[2], n_diff, (n_diff/tot)*100))
print("%23s: %4g (%3.1f%%)" %(LABELS[3], n_disc, (n_disc/tot)*100))

corrector = ClassCorrector(container=container)

#corrector.correct_class('arc', prediction_level=0, save_path=corrected_file)
corrector.correct_class('discrete', prediction_level=2, save_path=corrected_file)


''' Result for (b3 acc: 88%): datasets/Full_aurora_predicted_b3.json
entry.human_prediction == True (manual label)
tot:         3362
Aurora-less  1548 (46%)
Arc          307  (9%)
Diffuse      500  (15%)
Discrete     1007 (30%)

entry.human_prediction == False (predicted with b3)
tot:         4618
Aurora-less  2260 (49%)
Arc          533  (12%)
Diffuse      759  (16%)
Discrete     1066 (23%)

4618+3362=7980 (len(container))
'''
