#from lbl.class_corrector import ClassCorrector
from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer
import numpy as np
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------

LABELS = ['aurora-less', 'arc', 'diffuse', 'discrete']

predicted_file = r'C:\Users\Krist\Documents\ASI_json_files\Aurora_G_predicted_efficientnet-b2.json'

container = DatasetContainer.from_json(predicted_file)
print("len container: ", len(container))


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
            #plt.savefig("stats/b2/t_data/hour_plot_%s_%s.png" %(year, M_label[i]))
            plt.savefig("stats/Green/b2/hour_plot_%s_%s.png" %(year, M_label[i]))

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
    plt.savefig("stats/Green/b2//hour_plot_%s.png" %year)
    #plt.savefig("stats/b2/t_data/hour_plot_%s.png" %year)


Hour_subplot(year="2014", month=True)
Hour_subplot(year="2020", month=True)

plt.show()
