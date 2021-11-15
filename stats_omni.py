#from lbl.class_corrector import ClassCorrector
from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
# -----------------------------------------------------------------------------

LABELS = ['aurora-less', 'arc', 'diffuse', 'discrete']

predicted_file = r'C:\Users\Krist\Documents\ASI_json_files\Aurora_G_predicted_efficientnet-b2.json'

container = DatasetContainer.from_json(predicted_file)
print("len container: ", len(container))

def max_min_mean(list, label):

    print("{}, max, min, mean:".format(label))
    print(np.max(list))
    print(np.min(list))
    print(np.mean(list))

def neg_pos(list_, label):

    neg_count = len(list(filter(lambda x: (x < 0), list_)))
    pos_count = len(list(filter(lambda x: (x >= 0), list_)))
    print("Nr. of entries [{}] with neg and pos Bz values".format(label))
    print("neg: %g [%3.2f%%]" %(neg_count, (neg_count/len(list_))*100))
    print("pos: %g [%3.2f%%]" %(pos_count, (pos_count/len(list_))*100))
    #return neg_count, pos_count

def omni():

    Bz_values_GSE = []
    #Bz_values_GSM = []
    Bz_labels = []

    Bz_a_less = []

    count99 = 0
    count99_aless = 0

    for entry in container:

        if entry.label != "aurora-less":

            #print(entry.solarwind['Bz, nT (GSE)'])
            #print(entry.solarwind['Bz, nT (GSM)'])
            #print(entry.label)
            #print(entry.solarwind)
            if entry.solarwind['Bz, nT (GSM)'] == 9999.99:
                count99 += 1
            else:
                Bz_values_GSE.append(entry.solarwind['Bz, nT (GSM)'])
                #Bz_values_GSM.append(entry.solarwind['Bz, nT (GSM)'])
                Bz_labels.append(entry.label)

        else:
            # aurora-less
            if entry.solarwind['Bz, nT (GSM)'] == 9999.99:
                count99_aless += 1
            else:
                Bz_a_less.append(entry.solarwind['Bz, nT (GSM)'])


    max_min_mean(list=Bz_values_GSE, label='Aurora')
    max_min_mean(list=Bz_a_less, label=LABELS[0])

    neg_pos(list_=Bz_values_GSE, label='Aurora')
    neg_pos(list_=Bz_a_less, label=LABELS[0])

    print("Nr of entries (aurora) with 9999.99 value:    ", count99)
    print("Nr of entries (no aurora) with 9999.99 value: ", count99_aless)

    """ GSE:
    len container:  284840
    Aurora, max, min, mean:
    25.38
    -16.9
    0.14504999523854875
    Aurora-less, max, min, mean:
    25.9
    -16.78
    -0.001794082802598419
    Aurora, neg:  68933
    Aurora, pos:  67580
    Aurora-less, neg:  68771
    Aurora-less, pos:  56998
    Nr of entries (aurora) with 9999.99 value:     8750
    Nr of entries (no aurora) with 9999.99 value:  13808
    """


def omni_ting():

    a_less = []
    arc = []
    diff = []
    disc = []

    count99 = 0
    count99_aless = 0

    for entry in container:

        if entry.label == LABELS[0]:
            if entry.solarwind['Bz, nT (GSE)'] != 9999.99:
                a_less.append(entry.solarwind['Bz, nT (GSE)'])
            else:
                count99_aless += 1
        elif entry.label == LABELS[1]:
            if entry.solarwind['Bz, nT (GSE)'] != 9999.99:
                arc.append(entry.solarwind['Bz, nT (GSE)'])
            else:
                count99 += 1
        elif entry.label == LABELS[2]:
            if entry.solarwind['Bz, nT (GSE)'] != 9999.99:
                diff.append(entry.solarwind['Bz, nT (GSE)'])
            else:
                count99 += 1
        elif entry.label == LABELS[3]:
            if entry.solarwind['Bz, nT (GSE)'] != 9999.99:
                disc.append(entry.solarwind['Bz, nT (GSE)'])
            else:
                count99 += 1

    max_min_mean(list=a_less, label=LABELS[0])
    max_min_mean(list=arc, label=LABELS[1])
    max_min_mean(list=diff, label=LABELS[2])
    max_min_mean(list=disc, label=LABELS[3])

    neg_pos(a_less, LABELS[0])
    neg_pos(arc, LABELS[1])
    neg_pos(diff, LABELS[2])
    neg_pos(disc, LABELS[3])

    print("Nr. of [{}] entries: {}".format(LABELS[0], len(a_less)))
    print("Nr. of [{}] entries: {}".format(LABELS[1], len(arc)))
    print("Nr. of [{}] entries: {}".format(LABELS[2], len(diff)))
    print("Nr. of [{}] entries: {}".format(LABELS[3], len(disc)))

    print("Nr of entries (aurora) with 9999.99 value:    ", count99)
    print("Nr of entries (no aurora) with 9999.99 value: ", count99_aless)

'''
omni()
print('----------')
omni_ting()
'''

"""
dates = []
for entry in container:
    #print(entry.timepoint)
    dates.append(entry.timepoint[:10])
print(dates)

from collections import Counter
dcounts = Counter(d[0] for d in dates)
for d, count in dcounts.items():
    print('The total defects for date {} is {}'.format(d, count))
"""


def distribution(container, labels, year=None):
    '''
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
    '''
    n_less = 0; n_arc = 0; n_diff = 0; n_disc = 0; f = 0; tot = 0

    for entry in container:

        if year:
            if entry.timepoint[:4] == year:
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

        else:
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

    if year:
        print(year)
    else:
        print("All years:")

    print("%23s: %g (%3.1f%%)" %('Total classified images', tot, (tot/len(container))*100))
    print("%23s: %4g (%3.1f%%)" %(labels[0], n_less, (n_less/tot)*100))
    print("%23s: %4g (%3.1f%%)" %(labels[1], n_arc, (n_arc/tot)*100))
    print("%23s: %4g (%3.1f%%)" %(labels[2], n_diff, (n_diff/tot)*100))
    print("%23s: %4g (%3.1f%%)" %(labels[3], n_disc, (n_disc/tot)*100))
    print("Nr. of labels other than classes: ", f)

#distribution(container, LABELS)
#distribution(container, LABELS, year='2014')
#distribution(container, LABELS, year='2020')

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

def stats_aurora(label, year=False, weight=False):
    """
    blabla
    """
    Years = []
    Month = []
    Clock = []
    Hours = []
    TH = []

    def data(entry, Years, Month, Clock, Hours, TH, weight=False):
        Years.append(entry.timepoint[:4])
        Month.append(entry.timepoint[5:7])
        Clock.append(entry.timepoint[-8:])
        Hours.append(entry.timepoint[-8:-6])
        TH.append(entry.timepoint[5:7]+entry.timepoint[-8:-6]) # Month, day and hour
        if weight:
            # Funker dette?
            TH.append(entry.timepoint[5:7]+entry.timepoint[-8:-6]) # Month, day and hour

    def autolabel(n):
        """
        Attach a text label above each bar displaying its height
        autolabel() from: https://stackoverflow.com/a/42498711
        """
        for i in n:
            height = i.get_height()
            ax.text(i.get_x() + i.get_width()/2., 1.01*height,
                    '%d' % int(height), ha='center', va='bottom')

    if year:

        for entry in container:
            if entry.timepoint[:4] == year:

                if weight:
                    if entry.score[entry.label] >= 0.8:
                        """
                        Hvis modell mer enn 80% sikker paa label
                        Vekt denne dobbelt
                        """

                        if label == LABELS[0] or LABELS[1] or LABELS[2] or LABELS[3]:
                            if entry.label == label:
                                data(entry, Years, Month, Clock, Hours, TH, weight=True)

                        # arc, diffuse and discrete aurora is counted as one
                        elif label == "aurora":
                            if entry.label != LABELS[0]:
                                data(entry, Years, Month, Clock, Hours, TH, weight=True)

                else:

                    if label == LABELS[0] or LABELS[1] or LABELS[2] or LABELS[3]:
                        if entry.label == label:
                            data(entry, Years, Month, Clock, Hours, TH)

                    # arc, diffuse and discrete aurora is counted as one
                    elif label == "aurora":
                        if entry.label != LABELS[0]:
                            data(entry, Years, Month, Clock, Hours, TH)

    else:
        for entry in container:

            if weight:
                if entry.score[entry.label] >= 0.8:
                    """
                    Hvis modell mer enn 80% sikker paa label
                    Vekt denne dobbelt
                    """

                    if label == LABELS[0] or LABELS[1] or LABELS[2] or LABELS[3]:
                        if entry.label == label:
                            data(entry, Years, Month, Clock, Hours, TH, weight=True)

                    if label == "aurora": # arc, diffuse and discrete aurora is counted as one
                        if entry.label != LABELS[0]:
                            #print(entry.timepoint) # YYYY-MM-DD hh:mm:ss
                            data(entry, Years, Month, Clock, Hours, TH, weight=True)

            else:
                if label == LABELS[0] or LABELS[1] or LABELS[2] or LABELS[3]:
                    if entry.label == label:
                        data(entry, Years, Month, Clock, Hours, TH)

                if label == "aurora": # arc, diffuse and discrete aurora is counted as one
                    if entry.label != LABELS[0]:
                        #print(entry.timepoint) # YYYY-MM-DD hh:mm:ss
                        data(entry, Years, Month, Clock, Hours, TH)


    return Years, Month, Clock, Hours, TH

def get_hour_count_per_month(TH, hours):

    TH_01 = []; TH_01_c = []; TH_01_c_N = []
    TH_02 = []; TH_02_c = []; TH_02_c_N = []
    TH_03 = []; TH_03_c = []#; TH_03_c_N = []
    TH_10 = []; TH_10_c = []; TH_10_c_N = []
    TH_11 = []; TH_11_c = []; TH_11_c_N = []
    TH_12 = []; TH_12_c = []; TH_12_c_N = []

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
        TH_01_c.append(TH_01.count(hours[i]))
        TH_02_c.append(TH_02.count(hours[i]))
        TH_03_c.append(TH_03.count(hours[i]))
        TH_10_c.append(TH_10.count(hours[i]))
        TH_11_c.append(TH_11.count(hours[i]))
        TH_12_c.append(TH_12.count(hours[i]))

    for i in range(len(hours)):
        TH_01_c_N.append((TH_01_c[i]/sum(TH_01_c))*100)
        TH_02_c_N.append((TH_02_c[i]/sum(TH_02_c))*100)
        #TH_03_c_N.append((TH_03_c[i]/sum(TH_03_c))*100)
        TH_10_c_N.append((TH_10_c[i]/sum(TH_10_c))*100)
        TH_11_c_N.append((TH_11_c[i]/sum(TH_11_c))*100)
        TH_12_c_N.append((TH_12_c[i]/sum(TH_12_c))*100)

    return TH_01_c, TH_02_c, TH_03_c, TH_10_c, TH_11_c, TH_12_c, TH_01_c_N, TH_02_c_N, TH_10_c_N, TH_11_c_N, TH_12_c_N

def plot(hours, list, label, year, month=None, monthly=False, axis=False):
    plt.plot(hours, list, '-.', label=label+' - '+year)
    if axis:
        plt.xlabel("Hour of the day")
    plt.ylabel("Percentage")
    plt.legend()
    '''
    if monthly:
        plt.title("Stats {}".format(month))
        #plt.savefig("stats/Green/b2/hour_lineplot_{}_{}.png".format(year, month))
    else:
        plt.title("Stats")
        #plt.savefig("stats/Green/b2/hour_lineplot_{}.png".format(year))
    '''
def plot_hourly_nor(hours, T_c_N, T_arc_N, T_diff_N, T_disc_N, year, month=None, monthly=False):

    plt.figure()
    plt.plot(hours, T_c_N, '-.', label='No aurora')
    plt.plot(hours, T_arc_N, '-*', label='Arc')
    plt.plot(hours, T_diff_N, '-o', label='Diffuse')
    plt.plot(hours, T_disc_N, '-d', label='Discrete')
    plt.axhline(y = 5, color = 'silver', linestyle = '--')
    plt.axhline(y = 10, color = 'silver', linestyle = '--')
    #plt.xticks(rotation='vertical')
    plt.xlabel("Hour of the day"); plt.ylabel("Percentage")
    plt.legend()
    #plt.savefig("stats/Green/b2//hour_lineplot_per_%s.png" %year)
    if monthly:
        plt.title("Stats {} {}".format(month, year))
        plt.savefig("stats/Green/b2/hour_lineplot_{}_{}.png".format(year, month))
    else:
        plt.title("Stats {}".format(year))
        plt.savefig("stats/Green/b2/hour_lineplot_{}.png".format(year))

def Hour_subplot(year, month=False):

    hours = Get_hours()

    Years, Months, Clock, Hours, TH = stats_aurora(label="aurora-less", year=year, weight=False)
    Years_arc, Months_arc, Clock_arc, Hours_arc, TH_arc = stats_aurora(label="arc", year=year, weight=False)
    Years_diff, Months_diff, Clock_diff, Hours_diff, TH_diff = stats_aurora(label="diffuse", year=year, weight=False)
    Years_disc, Months_disc, Clock_disc, Hours_disc, TH_disc = stats_aurora(label="discrete", year=year, weight=False)

    T_c = []; T_arc = []; T_diff = []; T_disc = []
    T_c_N = []; T_arc_N = []; T_diff_N = []; T_disc_N = []

    if month:

        # Note, removed Mars, because of no data
        # Aurora-less
        TH_01, TH_02, TH_03, TH_10, TH_11, TH_12, \
        TH_01_N, TH_02_N, TH_10_N, TH_11_N, TH_12_N \
        = get_hour_count_per_month(TH, hours)
        # Arc
        TH_01_arc, TH_02_arc, TH_03_arc, TH_10_arc, TH_11_arc, TH_12_arc, \
        TH_01_arc_N, TH_02_arc_N, TH_10_arc_N, TH_11_arc_N, TH_12_arc_N \
        = get_hour_count_per_month(TH_arc, hours)
        # Diff
        TH_01_diff, TH_02_diff, TH_03_diff, TH_10_diff, TH_11_diff, TH_12_diff, \
        TH_01_diff_N, TH_02_diff_N, TH_10_diff_N, TH_11_diff_N, TH_12_diff_N \
        = get_hour_count_per_month(TH_diff, hours)
        # Disc
        TH_01_disc, TH_02_disc, TH_03_disc, TH_10_disc, TH_11_disc, TH_12_disc, \
        TH_01_disc_N, TH_02_disc_N, TH_10_disc_N, TH_11_disc_N, TH_12_disc_N \
        = get_hour_count_per_month(TH_disc, hours)

        T_c.extend([TH_01, TH_02, TH_03, TH_10, TH_11, TH_12])
        T_arc.extend([TH_01_arc, TH_02_arc, TH_03_arc, TH_10_arc, TH_11_arc, TH_12_arc])
        T_diff.extend([TH_01_diff, TH_02_diff, TH_03_diff, TH_10_diff, TH_11_diff, TH_12_diff])
        T_disc.extend([TH_01_disc, TH_02_disc, TH_03_disc, TH_10_disc, TH_11_disc, TH_12_disc])

        # Normalized
        T_c_N.extend([TH_01_N, TH_02_N, TH_10_N, TH_11_N, TH_12_N])
        T_arc_N.extend([TH_01_arc_N, TH_02_arc_N, TH_10_arc_N, TH_11_arc_N, TH_12_arc_N])
        T_diff_N.extend([TH_01_diff_N, TH_02_diff_N, TH_10_diff_N, TH_11_diff_N, TH_12_diff_N])
        T_disc_N.extend([TH_01_disc_N, TH_02_disc_N, TH_10_disc_N, TH_11_disc_N, TH_12_disc_N])


        M_ = ['01', '02', '03', '10', '11', '12']
        M_label = ['Jan', 'Feb', 'Mar', 'Oct', 'Nov', 'Dec']
        M_label_N = ['Jan', 'Feb', 'Oct', 'Nov', 'Dec']

        for i in range(len(M_label_N)):
            plot_hourly_nor(hours, T_c_N[i], T_arc_N[i], T_diff_N[i], T_disc_N[i], year, M_label_N[i], monthly=True)

        M_plots = False

        if M_plots:
            for i in range(len(M_)):

                plt.figure()
                subcategorybar(hours, [T_arc[i], T_diff[i], T_disc[i]], ["arc. tot: %s" %sum(T_arc[i]), "diff. tot: %s"%sum(T_diff[i]), "disc. tot: %s"%sum(T_disc[i])])
                plt.title("Stats %s, %s" %(M_label[i], year))
                #plt.xticks(rotation='vertical')
                plt.xlabel("Hour of the day"); plt.ylabel("Count")
                #plt.savefig("stats/Green/b2/hour_plot_%s_%s.png" %(year, M_label[i]))

    # Year
    T_c = []; T_arc = []; T_diff = []; T_disc = []
    T_c_N = []; T_arc_N = []; T_diff_N = []; T_disc_N = []

    for i in range(len(hours)):
        T_c.append(Hours.count(hours[i]))
        T_arc.append(Hours_arc.count(hours[i]))
        T_diff.append(Hours_diff.count(hours[i]))
        T_disc.append(Hours_disc.count(hours[i]))

    for i in range(len(hours)):
        T_c_N.append((T_c[i]/sum(T_c))*100)
        T_arc_N.append((T_arc[i]/sum(T_arc))*100)
        T_diff_N.append((T_diff[i]/sum(T_diff))*100)
        T_disc_N.append((T_disc[i]/sum(T_disc))*100)

    '''
    per = True
    if per:

        plt.figure()
        subcategorybar(hours, [T_arc_N, T_diff_N, T_disc_N], ["arc. tot: %d" %sum(T_arc), "diff. tot: %d"%sum(T_diff), "disc. tot: %d"%sum(T_disc)])
        plt.title("Stats %s" %year)
        #plt.xticks(rotation='vertical')
        plt.xlabel("Hour of the day"); plt.ylabel("Percentage")
        #plt.savefig("stats/Green/b2//hour_plot_per_%s.png" %year)
    else:
        plt.figure()
        subcategorybar(hours, [T_arc, T_diff, T_disc], ["arc. tot: %d" %sum(T_arc), "diff. tot: %d"%sum(T_diff), "disc. tot: %d"%sum(T_disc)])
        #%.1f?
        plt.title("Stats %s" %year)
        #plt.xticks(rotation='vertical')
        plt.xlabel("Hour of the day"); plt.ylabel("Count")
        #plt.savefig("stats/Green/b2//hour_plot_per_%s.png" %year)
    '''

    #figsize=(8, 6)
    if year == '2020':
        shape = '--'
    else:
        shape = '-'

    subplot(4,1,1)
    plt.title('Yearly statistics for all classes', fontsize=16)
    plt.plot(hours, T_arc_N, shape, label='arc - '+year)
    plt.ylabel("%", fontsize=13)
    plt.legend(fontsize=11)
    #plot(hours, T_arc_N, 'arc', year, month=None, monthly=False)

    subplot(4,1,2)
    #plot(hours, T_diff_N, 'diffuse', year, month=None, monthly=False)
    plt.plot(hours, T_diff_N, shape, label='diffuse - '+year)
    plt.ylabel("%", fontsize=13)
    plt.legend(fontsize=11)

    subplot(4,1,3)
    #plot(hours, T_disc_N, 'discrete', year, month=None, monthly=False)
    plt.plot(hours, T_disc_N, shape, label='discrete - '+year)
    plt.ylabel("%", fontsize=13)
    plt.legend(fontsize=11)

    subplot(4,1,4)
    #plot(hours, T_c_N, 'no aurora', year, month=None, monthly=False, axis=True)
    plt.plot(hours, T_c_N, shape, label='no aurora - '+year)
    plt.xlabel("Hour of the day", fontsize=13); plt.ylabel("%", fontsize=13)
    plt.legend(fontsize=11)

    #plot(hours, T_arc_N, 'arc', year, month=None, monthly=False)
    #plot(hours, T_diff_N, 'diffuse', year, month=None, monthly=False)
    #plot(hours, T_disc_N, 'discrete', year, month=None, monthly=False)
    #plot_hourly_nor(hours, T_c_N, T_arc_N, T_diff_N, T_disc_N, year)


Hour_subplot(year="2014", month=False)
Hour_subplot(year="2020", month=False)

#plt.savefig("stats/Green/b2/subs_all_classes.png")
plt.show()

'''
hours = 24

hours_year =
'''
