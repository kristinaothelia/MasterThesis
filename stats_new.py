from lbl.class_corrector import ClassCorrector
from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer
import numpy as np
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------

LABELS = ['aurora-less', 'arc', 'diffuse', 'discrete']

predicted_file_G = r'C:\Users\Krist\Documents\ASI_json_files\Aurora_G_predicted_efficientnet-b2.json'
predicted_file_R = r'C:\Users\Krist\Documents\ASI_json_files\Aurora_R_predicted_efficientnet-b2.json'

container_G = DatasetContainer.from_json(predicted_file_G)
container_R = DatasetContainer.from_json(predicted_file_R)
print("len container G: ", len(container_G))
print("len container R: ", len(container_R))

# Remove data for Feb, Mar, Oct
counter = 0
for i in range(len(container_G)):
    i -= counter
    if container_G[i].timepoint[5:7] == '02' \
    or container_G[i].timepoint[5:7] == '03' \
    or container_G[i].timepoint[5:7] == '10':
        del container_G[i]
        counter += 1
print('removed images from container_G: ', counter)
print('new container len: ', len(container_G))

def autolabel(n):
    """
    Attach a text label above each bar displaying its height
    autolabel() from: https://stackoverflow.com/a/42498711
    """
    for i in n:
        height = i.get_height()
        plt.text(i.get_x() + i.get_width()/2., 1.01*height,
                '%d' % int(height), ha='center', va='bottom')

def subcategorybar(X, vals, leg, width=0.8):
    """
    From: https://stackoverflow.com/a/48158449
    """
    n = len(vals)
    _X = np.arange(len(X))
    for i in range(n):
        #plt.bar(_X - width/2. + i/float(n)*width, vals[i],
        #        width=width/float(n), align="edge")
        autolabel(plt.bar(_X - width/2. + i/float(n)*width, vals[i],
                width=width/float(n), align="edge"))
    plt.legend(leg)
    plt.xticks(_X, X)

def distribution(container, labels):

    n_less = 0; n_arc = 0; n_diff = 0; n_disc = 0; f = 0; tot = 0

    for entry in container:

        while entry.label not in LABELS:
            print(entry.label)
            print(entry)
            f += 1

        if entry.human_prediction == False:  # False
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
    print("%23s: %4g (%3.1f%%)" %(labels[0], n_less, (n_less/tot)*100))
    print("%23s: %4g (%3.1f%%)" %(labels[1], n_arc, (n_arc/tot)*100))
    print("%23s: %4g (%3.1f%%)" %(labels[2], n_diff, (n_diff/tot)*100))
    print("%23s: %4g (%3.1f%%)" %(labels[3], n_disc, (n_disc/tot)*100))
    print("Nr. of labels other than classes: ", f)

#distribution(container, LABELS)

def pred_5050(container, title=''):

    weight = []
    dict = {}
    count_5050 = 0
    count_less_than_60 = 0
    count_over_80 = 0
    index = 0

    for entry in container:

        #print(entry.score[entry.label])
        #print(entry.score)
        index += 1

        if entry.score[entry.label] < 0.6:  # Less than 60 %

            #print("Max: %s " %entry.label, entry.score[entry.label])
            #print('Check second highest label')
            weight.append(1)

            x = sorted(entry.score.items(),key=(lambda i: i[1]))

            max = (x[-1][0], x[-1][1])
            max2nd = (x[-2][0], x[-2][1])

            if abs(max[1] - max2nd[1]) <= 0.1:

                count_5050 += 1
                '''
                print("max:    ", max)
                print("max2nd: ", max2nd)
                print()
                '''

                dict[index] = list()
                dict[index].extend([x[-1][0], x[-2][0]])

            else:
                dict[index] = None
                count_less_than_60 += 1
                #print("Not very similar max and max2nd, diff: ", abs(max[1] - max2nd[1]))
                #print("%.3f vs %.3f" %(max[1], max2nd[1]))

        else:
            weight.append(2)
            dict[index] = None
            #print("Max: %.5f [%s]" %(entry.score[entry.label], entry.label))
            if entry.score[entry.label] > 0.9:
                count_over_80 += 1


    #print(weight)
    print(title)
    #print(len(dict))
    print("Over 80% pred acc.: {} [{:.2f}%]".format(count_over_80, (count_over_80/len(dict))*100))
    print("50/50 labels: {} [{:.2f}%]".format(count_5050, (count_5050/len(dict))*100))
    print("(Less than 60% accuracy, but not 50/50: {})".format(count_less_than_60))

    c = 0
    test1 = 0
    test2 = 0
    test3 = 0
    test4 = 0
    test5 = 0
    test6 = 0


    for key, value in dict.items():
        if value != None:
            c += 1
            #print(value)

            if LABELS[0] in value and LABELS[1] in value:
                test1 += 1
            if LABELS[0] in value and LABELS[2] in value:
                test2 += 1
            if LABELS[0] in value and LABELS[3] in value:
                test3 += 1
            if LABELS[1] in value and LABELS[2] in value:
                test4 += 1
            if LABELS[1] in value and LABELS[3] in value:
                test5 += 1
            if LABELS[2] in value and LABELS[3] in value:
                test6 += 1

    print("count: %g, combi: %s (%3.1f%%)" %(test1, (LABELS[0], LABELS[1]), (test1/c)*100))
    print("count: %g, combi: %s (%3.1f%%)" %(test2, (LABELS[0], LABELS[2]), (test2/c)*100))
    print("count: %g, combi: %s (%3.1f%%)" %(test3, (LABELS[0], LABELS[3]), (test3/c)*100))
    print("count: %g, combi: %s (%3.1f%%)" %(test4, (LABELS[1], LABELS[2]), (test4/c)*100))
    print("count: %g, combi: %s (%3.1f%%)" %(test5, (LABELS[1], LABELS[3]), (test5/c)*100))
    print("count: %g, combi: %s (%3.1f%%)" %(test6, (LABELS[2], LABELS[3]), (test6/c)*100))
    #print(c)
    #print(test1+test2+test3+test4+test5+test6)

pred_5050(container_R, title='RED')
pred_5050(container_G, title='GREEN')

"""
Res for Aurora_R
bilder: 142470
Less than 60% accuracy, but not (ish) 50/50:  12158
ish 50/50 labels:  7315;

count: 899,  combi: ('aurora-less', 'arc')      (12.3%)
count: 1518, combi: ('aurora-less', 'diffuse')  (20.8%)
count: 227,  combi: ('aurora-less', 'discrete') (3.1%)
count: 545,  combi: ('arc', 'diffuse')          (7.5%)
count: 607,  combi: ('arc', 'discrete')         (8.3%)
count: 3519, combi: ('diffuse', 'discrete')     (48.1%)

tot: 7315

---

Res for Aurora_G
bilder: 284840
Less than 60% accuracy, but not (ish) 50/50:  22809
ish 50/50 labels:  13203;

count: 1851, combi: ('aurora-less', 'arc')      (14.0%)
count: 3382, combi: ('aurora-less', 'diffuse')  (25.6%)
count: 216,  combi: ('aurora-less', 'discrete') (1.6%)
count: 1935, combi: ('arc', 'diffuse')          (14.7%)
count: 1180, combi: ('arc', 'discrete')         (8.9%)
count: 4639, combi: ('diffuse', 'discrete')     (35.1%)

tot: 13203
"""
