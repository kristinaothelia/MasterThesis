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
'''
predicted_file = 't_data_with_2014nya4_predicted_b2.json'
#predicted_file = 'datasets/Full_aurora_predicted_b2.json'
#corrected_file = 'datasets/t_data_predicted_b2_corr.json'

container = DatasetContainer.from_json(predicted_file)
print("len container: ", len(container))
'''
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
    Clock = []
    Hours = []

    for entry in container:

        if year:
            if entry.timepoint[:4] == year:

                if label == LABELS[0] or LABELS[1] or LABELS[2] or LABELS[3]:
                    if entry.label == label:
                        Years.append(entry.timepoint[:4])
                        Month.append(entry.timepoint[5:7])
                        Clock.append(entry.timepoint[-8:])
                        Hours.append(entry.timepoint[-8:-6])

                if label == "aurora":
                    if entry.label != LABELS[0]:
                        Years.append(entry.timepoint[:4])
                        Month.append(entry.timepoint[5:7])
                        Clock.append(entry.timepoint[-8:])
                        Hours.append(entry.timepoint[-8:-6])

                else:
                    # Only predictions made by the model
                    if entry.human_prediction == False:
                        Years.append(entry.timepoint[:4])
                        Month.append(entry.timepoint[5:7])
                        Clock.append(entry.timepoint[-8:])
                        Hours.append(entry.timepoint[-8:-6])
        else:
            if label:
                if label == LABELS[0] or LABELS[1] or LABELS[2] or LABELS[3]:
                    if entry.label == label:
                        Years.append(entry.timepoint[:4])
                        Month.append(entry.timepoint[5:7])
                        Clock.append(entry.timepoint[-8:])
                        Hours.append(entry.timepoint[-8:-6])

                if label == "aurora":
                    if entry.label != LABELS[0]:
                        Years.append(entry.timepoint[:4])
                        Month.append(entry.timepoint[5:7])
                        Clock.append(entry.timepoint[-8:])
                        Hours.append(entry.timepoint[-8:-6])

            else:
                # Only predictions made by the model
                if entry.human_prediction == False:
                    Years.append(entry.timepoint[:4])
                    Month.append(entry.timepoint[5:7])
                    Clock.append(entry.timepoint[-8:])
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
        #plt.savefig("stats/b2/t_data/month_plot_%s.png" %year)


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

#stats(label="aurora", year="2014")
#stats(label="aurora", year="2020")

# Lage automatisk lagring
# Lage to soyler, aurora, ikke aurora.
# Lage 4 soyler, alle klassene? For month plot

#stats(label="aurora", year="2020")

#plt.show()

def stats_aurora(label, year=False):
    """
    blabla
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
    Clock = []
    Hours = []
    TH = []

    if year:

        for entry in container:
            if entry.timepoint[:4] == year:

                if label == LABELS[0] or LABELS[1] or LABELS[2] or LABELS[3]:
                    if entry.label == label:
                        Years.append(entry.timepoint[:4])
                        Month.append(entry.timepoint[5:7])
                        Clock.append(entry.timepoint[-8:])
                        Hours.append(entry.timepoint[-8:-6])
                        TH.append(entry.timepoint[5:7]+entry.timepoint[-8:-6]) # Month, day and hour

                # arc, diffuse and discrete aurora is counted as one
                elif label == "aurora":
                    if entry.label != LABELS[0]:
                        Years.append(entry.timepoint[:4])
                        Month.append(entry.timepoint[5:7])
                        Clock.append(entry.timepoint[-8:])
                        Hours.append(entry.timepoint[-8:-6])
                        TH.append(entry.timepoint[5:7]+entry.timepoint[-8:-6]) # Month, day and hour

    else:

        for entry in container:
            if label == LABELS[0] or LABELS[1] or LABELS[2] or LABELS[3]:
                if entry.label == label:
                    Years.append(entry.timepoint[:4])
                    Month.append(entry.timepoint[5:7])
                    Clock.append(entry.timepoint[-8:])
                    Hours.append(entry.timepoint[-8:-6])
                    TH.append(entry.timepoint[5:7]+entry.timepoint[-8:-6]) # Month, day and hour

            if label == "aurora": # arc, diffuse and discrete aurora is counted as one
                if entry.label != LABELS[0]:
                    #print(entry.timepoint) # YYYY-MM-DD hh:mm:ss
                    Years.append(entry.timepoint[:4])
                    Month.append(entry.timepoint[5:7])
                    Clock.append(entry.timepoint[-8:])
                    Hours.append(entry.timepoint[-8:-6])
                    TH.append(entry.timepoint[5:7]+entry.timepoint[-8:-6]) # Month, day and hour

            # If NO label provided
            else:
                print("data not correctly labeled")
                exit()

    '''
    for entry in container:

        # If year provided
        if year:
            if entry.timepoint[:4] == year:

                # All 4 labels
                while entry.label in LABELS:
                #if label == LABELS[0] or LABELS[1] or LABELS[2] or LABELS[3]:
                    if entry.label == label:
                        Years.append(entry.timepoint[:4])
                        Month.append(entry.timepoint[5:7])
                        Clock.append(entry.timepoint[-8:])
                        Hours.append(entry.timepoint[-8:-6])
                        TH.append(entry.timepoint[5:7]+entry.timepoint[-8:-6]) # Month, day and hour

                # arc, diffuse and discrete aurora is counted
                if label == "aurora":
                    if entry.label != LABELS[0]:
                        Years.append(entry.timepoint[:4])
                        Month.append(entry.timepoint[5:7])
                        Clock.append(entry.timepoint[-8:])
                        Hours.append(entry.timepoint[-8:-6])
                        TH.append(entry.timepoint[5:7]+entry.timepoint[-8:-6]) # Month, day and hour

                # If NO label provided
                else:
                    print("data not correctly labeled")
                    exit()

        # If year NOT provided
        else:
            if label == LABELS[0] or LABELS[1] or LABELS[2] or LABELS[3]:
                if entry.label == label:
                    Years.append(entry.timepoint[:4])
                    Month.append(entry.timepoint[5:7])
                    Clock.append(entry.timepoint[-8:])
                    Hours.append(entry.timepoint[-8:-6])
                    TH.append(entry.timepoint[5:7]+entry.timepoint[-8:-6]) # Month, day and hour

            if label == "aurora": # arc, diffuse and discrete aurora is counted
                if entry.label != LABELS[0]:
                    #print(entry.timepoint) # YYYY-MM-DD hh:mm:ss
                    Years.append(entry.timepoint[:4])
                    Month.append(entry.timepoint[5:7])
                    Clock.append(entry.timepoint[-8:])
                    Hours.append(entry.timepoint[-8:-6])
                    TH.append(entry.timepoint[5:7]+entry.timepoint[-8:-6]) # Month, day and hour

            # If NO label provided
            else:
                print("data not correctly labeled")
                exit()

    '''
    return Years, Month, Clock, Hours, TH

def get_hour_count_per_month(TH, hours):

    TH_01 = []; TH_01_count = []
    TH_02 = []; TH_02_count = []
    TH_03 = []; TH_03_count = []
    TH_10 = []; TH_10_count = []
    TH_11 = []; TH_11_count = []
    TH_12 = []; TH_12_count = []

    for i in range(len(TH)):
        if TH[i][:2] == "01":
            TH_01.append(TH[i][-2:])
        elif TH[i][:2] == "02":
            TH_02.append(TH[i][-2:])
        elif TH[i][:2] == "03":
            TH_03.append(TH[i][-2:])
        elif TH[i][:2] == "10":
            TH_10.append(TH[i][-2:])
        elif TH[i][:2] == "11":
            TH_11.append(TH[i][-2:])
        elif TH[i][:2] == "12":
            TH_12.append(TH[i][-2:])

    for i in range(len(hours)):
        TH_01_count.append(TH_01.count(hours[i]))
        TH_02_count.append(TH_02.count(hours[i]))
        TH_03_count.append(TH_03.count(hours[i]))
        TH_10_count.append(TH_10.count(hours[i]))
        TH_11_count.append(TH_11.count(hours[i]))
        TH_12_count.append(TH_12.count(hours[i]))

    return TH_01_count, TH_02_count, TH_03_count, TH_10_count, TH_11_count, TH_12_count


'''
Years_14, Months_14, Clock_14, Hours_14 = stats_aurora(label="aurora-less", year="2014")
Years_arc_14, Months_arc_14, Clock_arc_14, Hours_arc_14 = stats_aurora(label="arc", year="2014")
Years_diff_14, Months_diff_14, Clock_diff_14, Hours_diff_14 = stats_aurora(label="diffuse", year="2014")
Years_disc_14, Months_disc_14, Clock_disc_14, Hours_disc_14 = stats_aurora(label="discrete", year="2014")

Years_20, Months_20, Clock_20, Hours_20 = stats_aurora(label="aurora-less", year="2020")
Years_arc_20, Months_arc_20, Clock_arc_20, Hours_arc_20 = stats_aurora(label="arc", year="2020")
Years_diff_20, Months_diff_20, Clock_diff_20, Hours_diff_20 = stats_aurora(label="diffuse", year="2020")
Years_disc_20, Months_disc_20, Clock_disc_20, Hours_disc_20 = stats_aurora(label="discrete", year="2020")


M_ = ['01', '02', '03', '10', '11', '12']
M_label = ['Jan', 'Feb', 'Mar', 'Oct', 'Nov', 'Dec']

M_c = [Months_20.count(M_[0]), Months_20.count(M_[1]), Months_20.count(M_[2]),
     Months_20.count(M_[3]), Months_20.count(M_[4]), Months_20.count(M_[5])]

M_arc = [Months_arc_20.count(M_[0]), Months_arc_20.count(M_[1]), Months_arc_20.count(M_[2]),
     Months_arc_20.count(M_[3]), Months_arc_20.count(M_[4]), Months_arc_20.count(M_[5])]

M_diff = [Months_diff_20.count(M_[0]), Months_diff_20.count(M_[1]), Months_diff_20.count(M_[2]),
     Months_diff_20.count(M_[3]), Months_diff_20.count(M_[4]), Months_diff_20.count(M_[5])]

M_disc = [Months_disc_20.count(M_[0]), Months_disc_20.count(M_[1]), Months_disc_20.count(M_[2]),
     Months_disc_20.count(M_[3]), Months_disc_20.count(M_[4]), Months_disc_20.count(M_[5])]

plt.figure()
subcategorybar(M_label, [M_arc, M_diff, M_disc, M_c], ["arc", "diff", "disc", "no aurora"])
plt.title("Stats for Jan, Feb, Mar, Oct, Nov, Dec, 2020")
#plt.xticks(rotation='vertical')
plt.xlabel("Month"); plt.ylabel("Count")
#plt.show()
'''

def Hour_subplot(year, month=False):

    hours = Get_hours()

    Years, Months, Clock, Hours, TH = stats_aurora(label="aurora-less", year=year)
    Years_arc, Months_arc, Clock_arc, Hours_arc, TH_arc = stats_aurora(label="arc", year=year)
    Years_diff, Months_diff, Clock_diff, Hours_diff, TH_diff = stats_aurora(label="diffuse", year=year)
    Years_disc, Months_disc, Clock_disc, Hours_disc, TH_disc = stats_aurora(label="discrete", year=year)

    T_c = []; T_arc = []; T_diff = []; T_disc = []

    if month:

        TH_01, TH_02, TH_03, TH_10, TH_11, TH_12 = get_hour_count_per_month(TH, hours)
        TH_01_arc, TH_02_arc, TH_03_arc, TH_10_arc, TH_11_arc, TH_12_arc = get_hour_count_per_month(TH_arc, hours)
        TH_01_diff, TH_02_diff, TH_03_diff, TH_10_diff, TH_11_diff, TH_12_diff = get_hour_count_per_month(TH_diff, hours)
        TH_01_disc, TH_02_disc, TH_03_disc, TH_10_disc, TH_11_disc, TH_12_disc = get_hour_count_per_month(TH_disc, hours)

        T_c.extend([TH_01, TH_02, TH_03, TH_10, TH_11, TH_12])
        T_arc.extend([TH_01_arc, TH_02_arc, TH_03_arc, TH_10_arc, TH_11_arc, TH_12_arc])
        T_diff.extend([TH_01_diff, TH_02_diff, TH_03_diff, TH_10_diff, TH_11_diff, TH_12_diff])
        T_disc.extend([TH_01_disc, TH_02_disc, TH_03_disc, TH_10_disc, TH_11_disc, TH_12_disc])


        M_ = ['01', '02', '03', '10', '11', '12']
        M_label = ['Jan', 'Feb', 'Mar', 'Oct', 'Nov', 'Dec']

        for i in range(len(M_)):

            plt.figure()
            subcategorybar(hours, [T_arc[i], T_diff[i], T_disc[i]], ["arc. tot: %s" %sum(T_arc[i]), "diff. tot: %s"%sum(T_diff[i]), "disc. tot: %s"%sum(T_disc[i])])
            plt.title("Stats %s, %s" %(M_label[i], year))
            #plt.xticks(rotation='vertical')
            plt.xlabel("Hour of the day"); plt.ylabel("Count")
            plt.savefig("stats/b2/t_data/hour_plot_%s_%s.png" %(year, M_label[i]))

    # Year
    T_c = []; T_arc = []; T_diff = []; T_disc = []

    for i in range(len(hours)):
        T_c.append(Hours.count(hours[i]))
        T_arc.append(Hours_arc.count(hours[i]))
        T_diff.append(Hours_diff.count(hours[i]))
        T_disc.append(Hours_disc.count(hours[i]))

    plt.figure()
    subcategorybar(hours, [T_arc, T_diff, T_disc], ["arc. tot: %d" %sum(T_arc), "diff. tot: %d"%sum(T_diff), "disc. tot: %d"%sum(T_disc)])
    plt.title("Stats %s" %year)
    #plt.xticks(rotation='vertical')
    plt.xlabel("Hour of the day"); plt.ylabel("Count")
    plt.savefig("stats/b2/t_data/hour_plot_%s.png" %year)


#Hour_subplot(year="2014", month=True)
#Hour_subplot(year="2020", month=True)

#plt.show()


'''
Years, Month, Clock, Hours, TH = stats_aurora(label="aurora")
hours = Get_hours()
TH_01_count, TH_02_count, TH_03_count, TH_10_count, TH_11_count, TH_12_count = get_hour_count_per_month(TH, hours)


plt.figure()
subcategorybar(hours, [TH_01_count, TH_02_count], ["Jan", "Feb"])
plt.title("Stats")
#plt.xticks(rotation='vertical')
plt.xlabel("Hour of the day"); plt.ylabel("Count")
plt.show()
'''


# Stats showing different aurora per hour, for a month














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
