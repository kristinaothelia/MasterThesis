from lbl.class_corrector import ClassCorrector
from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer
import numpy as np
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------

"""
Some statistics for the classified data, 4 labels

eks fra json fil:
"timepoint": "2015-11-17 18:18:37",
"label": null,
"human_prediction": null,
"score": {}
"""

LABELS = ['aurora-less', 'arc', 'diffuse', 'discrete']

predicted_file = 't_data_with_2014nya4_predicted_b2.json'
#predicted_file = 'datasets/Full_aurora_predicted_b2.json'
#corrected_file = 'datasets/t_data_predicted_b2_corr.json'

container = DatasetContainer.from_json(predicted_file)

def distribution(container, labels):

    n_less = 0; n_arc = 0; n_diff = 0; n_disc = 0; tot = 0

    for entry in container:

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

distribution(container, LABELS)

def Get_hours():
    # Make times
    times = []
    for i in range(24):

        if i < 10:
            str = '0%s' %i
        else:
            str = '%s' %i
        times.append(str)
    return times


def stats(label=False, year=False, plot_year=True):
    """
    label and year needs to be string input; "arc", "2020"
    """

    def autolabel(n):
        """
        Attach a text label above each bar displaying its height
        autolabel() from: https://stackoverflow.com/a/42498711
        """
        for i in n:
            height = i.get_height()
            ax.text(i.get_x() + i.get_width()/2., 1.01*height,
                    '%d' % int(height), ha='center', va='bottom')

    Years = []
    Month = []
    Times = []
    Hours = []

    for entry in container:

        if year:
            if entry.timepoint[:4] == year:

                if label:
                    if entry.label == label:

                        Years.append(entry.timepoint[:4])
                        Month.append(entry.timepoint[5:7])
                        Times.append(entry.timepoint[-8:])
                        Hours.append(entry.timepoint[-8:-6])
                else:
                    # Only predictions made by the model
                    if entry.human_prediction == False:

                        Years.append(entry.timepoint[:4])
                        Month.append(entry.timepoint[5:7])
                        Times.append(entry.timepoint[-8:])
                        Hours.append(entry.timepoint[-8:-6])
        else:
            if label:
                if entry.label == label:

                    Years.append(entry.timepoint[:4])
                    Month.append(entry.timepoint[5:7])
                    Times.append(entry.timepoint[-8:])
                    Hours.append(entry.timepoint[-8:-6])
            else:
                # Only predictions made by the model
                if entry.human_prediction == False:

                    Years.append(entry.timepoint[:4])
                    Month.append(entry.timepoint[5:7])
                    Times.append(entry.timepoint[-8:])
                    Hours.append(entry.timepoint[-8:-6])

    if label == False:
        label_plot = "All classes"
    else:
        label_plot = label

    if year == False:
        year_plot = "2014 and 2020"

    if year:

        if plot_year:
            Y_ = year
            Y = Years.count(Y_)

            fig, ax = plt.subplots()
            ny = ax.bar(Y_, Y)
            autolabel(ny)
            plt.title("%s: %s" %(label_plot, year))
            #plt.xticks(rotation=70)
            plt.xlabel("Year"); plt.ylabel("Count")


        M_ = ['01', '02', '03', '10', '11', '12']
        M_label = ['Jan', 'Feb', 'Mar', 'Oct', 'Nov', 'Dec']
        M = [Month.count(M_[0]), Month.count(M_[1]), Month.count(M_[2]),
             Month.count(M_[3]), Month.count(M_[4]), Month.count(M_[5])]

        fig, ax = plt.subplots()
        n = ax.bar(M_label, M)
        autolabel(n)
        plt.title("%s %s: for Jan, Feb, Mar, Oct, Nov, Dec" %(label_plot, year))
        #plt.xticks(rotation='vertical')
        plt.xlabel("Month"); plt.ylabel("Count")


        hours = Get_hours()
        T = []
        for i in range(len(hours)):
            T.append(Hours.count(hours[i]))

        fig, ax = plt.subplots()
        h = ax.bar(hours, T)
        autolabel(h)
        plt.title("%s: %s" %(label_plot, year))
        plt.xlabel("Hour of the day"); plt.ylabel("Count")

    else:

        Y_ = ['2014', '2020']
        Y = [Years.count(Y_[0]), Years.count(Y_[1])]

        fig, ax = plt.subplots()
        ny = ax.bar(Y_, Y)
        autolabel(ny)
        plt.title("%s: %s" %(label_plot, year_plot))
        #plt.xticks(rotation=70)
        plt.xlabel("Year"); plt.ylabel("Count")

        M_ = ['01', '02', '03', '10', '11', '12']
        M_label = ['Jan', 'Feb', 'Mar', 'Oct', 'Nov', 'Dec']
        M = [Month.count(M_[0]), Month.count(M_[1]), Month.count(M_[2]),
             Month.count(M_[3]), Month.count(M_[4]), Month.count(M_[5])]

        fig, ax = plt.subplots()
        n = ax.bar(M_label, M)
        autolabel(n)
        plt.title("%s %s: for Jan, Feb, Mar, Oct, Nov, Dec" %(label_plot, year_plot))
        #plt.xticks(rotation='vertical')
        plt.xlabel("Month"); plt.ylabel("Count")

        hours = Get_hours()
        T = []
        for i in range(len(hours)):
            T.append(Hours.count(hours[i]))

        fig, ax = plt.subplots()
        h = ax.bar(hours, T)
        autolabel(h)
        plt.title("%s: %s" %(label_plot, year_plot))
        plt.xlabel("Hour of the day"); plt.ylabel("Count")


#stats(label='arc', year="2014")
#stats(label='arc', year="2020")
#stats(label="discrete", year="2014", plot_year=False)
#stats(label="discrete", year="2020", plot_year=False)
stats(year="2014")
stats(year="2020")
plt.show()





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

'''
