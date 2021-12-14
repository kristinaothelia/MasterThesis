#from lbl.class_corrector import ClassCorrector
from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from pylab import *
# -----------------------------------------------------------------------------

# Red:      6300        Count:  142 470
# Green:    5577        Count:  284 840

LABELS = ['aurora-less', 'arc', 'diffuse', 'discrete']

# All 4 years, jan+nov+dec
predicted_G_Full = r'C:\Users\Krist\Documents\ASI_json_files\AuroraFull_G_omni_mean_predicted_efficientnet-b2.json'
container_Full = DatasetContainer.from_json(predicted_G_Full)
print("len container Full: ", len(container_Full))

split_day = False
if split_day:
    container_D = DatasetContainer.from_json(r'C:\Users\Krist\Documents\ASI_json_files\AuroraFull_G_omni_mean_predicted_efficientnet-b2_daytime.json')
    container_N = DatasetContainer.from_json(r'C:\Users\Krist\Documents\ASI_json_files\AuroraFull_G_omni_mean_predicted_efficientnet-b2_nighttime.json')

    print('len container day:   ', len(container_D))
    print('len container night: ', len(container_N))

# make distribution based on predictions
def distribution(container, labels, year=None, month=False):    #, print=False

    n_less = 0; n_arc = 0; n_diff = 0; n_disc = 0; f = 0; tot = 0

    jan = 0;     nov = 0;     dec = 0
    jan_arc = 0;     nov_arc = 0;     dec_arc = 0
    jan_diff = 0;     nov_diff = 0;     dec_diff = 0
    jan_disc = 0;     nov_disc = 0;     dec_disc = 0

    def M(entry, jan, nov, dec):
        if entry.timepoint[5:7] == '01':
            jan += 1
        if entry.timepoint[5:7] == '11':
            nov += 1
        if entry.timepoint[5:7] == '12':
            dec += 1

        return jan, nov, dec

    for entry in container:

        if year:
            year = year[:4]
            if month:
                if entry.timepoint[:4] == year:
                    tot += 1
                    if entry.label == LABELS[1]:
                        n_arc += 1
                        jan_arc, nov_arc, dec_arc \
                        = M(entry, jan_arc, nov_arc, dec_arc)
                    elif entry.label == LABELS[2]:
                        n_diff += 1
                        jan_diff, nov_diff, dec_diff \
                        = M(entry, jan_diff, nov_diff, dec_diff)
                    elif entry.label == LABELS[3]:
                        n_disc += 1
                        jan_disc, nov_disc, dec_disc \
                        = M(entry, jan_disc, nov_disc, dec_disc)
                    else:
                        n_less += 1
                        jan, nov, dec = M(entry, jan, nov, dec)

            else:
                if entry.timepoint[:4] == year:
                    #if entry.human_prediction == False:  # False
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
            if month:
                if entry.timepoint[:4] == year:
                    tot += 1
                    if entry.label == LABELS[1]:
                        n_arc += 1
                        jan_arc, nov_arc, dec_arc = M(entry, jan, nov, dec)
                    elif entry.label == LABELS[2]:
                        n_diff += 1
                        jan_diff, nov_diff, dec_diff = M(entry, jan, nov, dec)
                    elif entry.label == LABELS[3]:
                        n_disc += 1
                        jan_disc, nov_disc, dec_disc = M(entry, jan, nov, dec)
                    else:
                        n_less += 1
                        jan, nov, dec = M(entry, jan, nov, dec)

            else:
                #if entry.human_prediction == False:  # False
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

    if month:
        print('sum jan: ', sum([jan, jan_arc, jan_diff, jan_disc]))
        print('sum nov: ', sum([nov, nov_arc, nov_diff, nov_disc]))
        print('sum dec: ', sum([dec, dec_arc, dec_diff, dec_disc]))

    n_less_M = []
    n_arc_M = []
    n_diff_M = []
    n_disc_M = []

    n_less_M.extend([jan, nov, dec])
    n_arc_M.extend([jan_arc, nov_arc, dec_arc])
    n_diff_M.extend([jan_diff, nov_diff, dec_diff])
    n_disc_M.extend([jan_disc, nov_disc, dec_disc])

    return tot, n_less, n_arc, n_diff, n_disc, n_less_M, n_arc_M, n_diff_M, n_disc_M


#distribution(container_Full, LABELS, year='2014', print=True)
#distribution(container_Full, LABELS, year='2016', print=True)
#distribution(container_Full, LABELS, year='2018', print=True)
#distribution(container_Full, LABELS, year='2020', print=True)


# make distribution pie charts
def month_subpie(for_year, n_less, n_arc, n_diff, n_disc, labels, title='', a_less_plot=True):

    if a_less_plot == False:
        jan = [n_arc[0], n_diff[0], n_disc[0]]
        nov = [n_arc[1], n_diff[1], n_disc[1]]
        dec = [n_arc[2], n_diff[2], n_disc[2]]
        for_year = for_year[1:]
        labels = labels[1:]

        explode = (0.05, 0.05, 0.05)
        colors = ['dodgerblue','forestgreen', 'mediumslateblue']
    else:
        jan = [n_less[0], n_arc[0], n_diff[0], n_disc[0]]
        nov = [n_less[1], n_arc[1], n_diff[1], n_disc[1]]
        dec = [n_less[2], n_arc[2], n_diff[2], n_disc[2]]

        explode = (0.05, 0.05, 0.05, 0.05)
        colors = ['dimgrey','dodgerblue','forestgreen', 'mediumslateblue']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(7, 8))
    fig.suptitle("Distribution for {}".format(title), fontsize=16)
    angle = 90

    # jan
    ax1.pie(jan, explode=explode, labels=labels, colors=colors, autopct = '%1.1f%%',
            shadow=True, textprops={'fontsize': 11}, startangle=angle)    #, startangle=90
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title("Jan", fontsize=13)

    # nov
    ax2.pie(nov, explode=explode, labels=labels, colors=colors, autopct = '%1.1f%%',
            shadow=True, textprops={'fontsize': 11}, startangle=angle)    #, startangle=90
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax2.set_title("Nov", fontsize=13)

    # dec
    ax3.pie(dec, explode=explode, labels=labels, colors=colors, autopct = '%1.1f%%',
            shadow=True, textprops={'fontsize': 11}, startangle=angle)    #, startangle=90
    ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax3.set_title("Dec", fontsize=13)
    #plt.legend(fontsize=11)

    # year
    ax4.pie(for_year, explode=explode, labels=labels, colors=colors, autopct = '%1.1f%%',
            shadow=True, textprops={'fontsize': 11}, startangle=angle)    #, startangle=90
    ax4.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax4.set_title("All months", fontsize=13)

    #plt.show()

def bar_chart(container, labels, year, wl, a_less=True, month=False):

    colors = ['dimgrey','dodgerblue','forestgreen', 'mediumslateblue']

    if month:
        for i in range(len(year)):

            tot, n_less, n_arc, n_diff, n_disc, \
            n_less_M, n_arc_M, n_diff_M, n_disc_M \
            = distribution(container, labels, year[i], month)

            for_year = [n_less, n_arc, n_diff, n_disc]

            # month bar chart

    else:
        tot1, n_less1, n_arc1, n_diff1, n_disc1, \
        n_less_M1, n_arc_M1, n_diff_M1, n_disc_M1 \
        = distribution(container, labels, year[0], month)

        tot2, n_less2, n_arc2, n_diff2, n_disc2, \
        n_less_M2, n_arc_M2, n_diff_M2, n_disc_M2 \
        = distribution(container, labels, year[1], month)

        tot3, n_less3, n_arc3, n_diff3, n_disc3, \
        n_less_M3, n_arc_M3, n_diff_M3, n_disc_M3 \
        = distribution(container, labels, year[2], month)

        tot4, n_less4, n_arc4, n_diff4, n_disc4, \
        n_less_M4, n_arc_M4, n_diff_M4, n_disc_M4 \
        = distribution(container, labels, year[3], month)

        if a_less:
            sizes1 = [n_less1/tot1, n_less2/tot2, n_less3/tot3, n_less4/tot4]
            sizes2 = [n_arc1/tot1, n_arc2/tot2, n_arc3/tot3, n_arc4/tot4]
            sizes3 = [n_diff1/tot1, n_diff2/tot2, n_diff3/tot3, n_diff4/tot4]
            sizes4 = [n_disc1/tot1, n_disc2/tot2, n_disc3/tot3, n_disc4/tot4]
        else:
            tot1 = tot1 - n_less1
            tot2 = tot2 - n_less2
            tot3 = tot3 - n_less3
            tot4 = tot4 - n_less4

            sizes1 = []
            sizes2 = [n_arc1/tot1, n_arc2/tot2, n_arc3/tot3, n_arc4/tot4]
            sizes3 = [n_diff1/tot1, n_diff2/tot2, n_diff3/tot3, n_diff4/tot4]
            sizes4 = [n_disc1/tot1, n_disc2/tot2, n_disc3/tot3, n_disc4/tot4]

        sizes1 = [element * 100 for element in sizes1]
        sizes2 = [element * 100 for element in sizes2]
        sizes3 = [element * 100 for element in sizes3]
        sizes4 = [element * 100 for element in sizes4]

        N = 4
        ind = np.arange(N)  # the x locations for the groups
        width = 0.22       # the width of the bars

        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)

        if a_less:
            yvals = sizes1
            rects1 = ax.bar(ind, yvals, width*0.9, color=colors[0], edgecolor = 'black')

            zvals = sizes2
            rects2 = ax.bar(ind+width, zvals, width*0.9, color=colors[1], edgecolor = 'black')

            kvals = sizes3
            rects3 = ax.bar(ind+width*2, kvals, width*0.9, color=colors[2], edgecolor = 'black')

            jvals = sizes4
            rects4 = ax.bar(ind+width*3, jvals, width*0.9, color=colors[3], edgecolor = 'black')

            plt.title(r"Distribution of predicted classes, {}".format(wl[0]), fontsize=16)
            ax.set_xticks(ind+width*1.5)
            ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), (r'no aurora', r'arc', r'diffuse', r'discrete'), fancybox=True, shadow=True, ncol=2, fontsize=11)
        else:
            zvals = sizes2
            rects2 = ax.bar(ind, zvals, width*0.9, color=colors[1], edgecolor = 'black')

            kvals = sizes3
            rects3 = ax.bar(ind+width, kvals, width*0.9, color=colors[2], edgecolor = 'black')

            jvals = sizes4
            rects4 = ax.bar(ind+width*2, jvals, width*0.9, color=colors[3], edgecolor = 'black')

            plt.title(r"Distribution of predicted aurora classes, {}".format(wl[0]), fontsize=16)
            ax.set_xticks(ind+width)
            ax.legend((rects2[0], rects3[0], rects4[0]), (r'arc', r'diffuse', r'discrete'), fancybox=True, shadow=True, ncol=1, fontsize=11)

        ax.set_ylabel(r'Normalized class count', fontsize=13)
        ax.set_xticklabels(year, fontsize=13)

        def autolabel(rects):
            for rect in rects:
                h = rect.get_height()
                ax.text(rect.get_x()+rect.get_width()/2., 1.5, '%.1f %%'%float(h),
                        ha='center', va='bottom', rotation=90) #1.05*h
        if a_less:
            autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        autolabel(rects4)

        plt.ylim(0, 55)
        plt.tight_layout()



def dist_pie_chart(container, labels, year, wl, a_less=True, month=False):

    explode = (0.05, 0.05, 0.05, 0.05)
    colors = ['dimgrey','dodgerblue','forestgreen', 'mediumslateblue']

    if month:
        for i in range(len(year)):

            tot, n_less, n_arc, n_diff, n_disc, \
            n_less_M, n_arc_M, n_diff_M, n_disc_M \
            = distribution(container, labels, year[i], month)

            for_year = [n_less, n_arc, n_diff, n_disc]

            month_subpie(for_year, n_less_M, n_arc_M, n_diff_M, n_disc_M, labels, year[i], a_less)

    else:

        tot1, n_less1, n_arc1, n_diff1, n_disc1, \
        n_less_M1, n_arc_M1, n_diff_M1, n_disc_M1 \
        = distribution(container, labels, year[0], month)

        tot2, n_less2, n_arc2, n_diff2, n_disc2, \
        n_less_M2, n_arc_M2, n_diff_M2, n_disc_M2 \
        = distribution(container, labels, year[1], month)

        tot3, n_less3, n_arc3, n_diff3, n_disc3, \
        n_less_M3, n_arc_M3, n_diff_M3, n_disc_M3 \
        = distribution(container, labels, year[2], month)

        tot4, n_less4, n_arc4, n_diff4, n_disc4, \
        n_less_M4, n_arc_M4, n_diff_M4, n_disc_M4 \
        = distribution(container, labels, year[3], month)

        #Pie chart, where the slices will be ordered and plotted counter-clockwise:
        sizes1 = [n_less1, n_arc1, n_diff1, n_disc1]
        sizes2 = [n_less2, n_arc2, n_diff2, n_disc2]
        sizes3 = [n_less3, n_arc3, n_diff3, n_disc3]
        sizes4 = [n_less4, n_arc4, n_diff4, n_disc4]


        if a_less == False:
            sizes1 = sizes1[1:]
            sizes2 = sizes2[1:]
            sizes3 = sizes3[1:]
            sizes4 = sizes4[1:]
            labels = labels[1:]
            colors = colors[1:]
            explode = explode[1:]
            tot1 = tot1 - n_less1
            tot2 = tot2 - n_less2
            tot3 = tot3 - n_less3
            tot4 = tot4 - n_less4

        title = "Distribution"
        if len(labels) == 3:
            title = "Aurora distribution"

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(6.5, 8.5))#,figsize=(8, 4.5)
        fig.suptitle(title, fontsize=16)

        ax1.pie(sizes1, explode=explode, labels=labels, colors=colors, autopct = '%1.1f%%',
                shadow=True, textprops={'fontsize': 11})    #, startangle=90
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax1.set_title("{} [{}]\n# of images: {}".format(year[0], wl[0], tot1), fontsize=13)

        ax2.pie(sizes2, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, textprops={'fontsize': 11})#, startangle=90
        ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax2.set_title("{} [{}]\n# of images: {}".format(year[1], wl[1], tot2), fontsize=13)

        ax3.pie(sizes3, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, textprops={'fontsize': 11})#, startangle=90
        ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax3.set_title("{} [{}]\n# of images: {}".format(year[2], wl[2], tot3), fontsize=13)

        ax4.pie(sizes4, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, textprops={'fontsize': 11})#, startangle=90
        ax4.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax4.set_title("{} [{}]\n# of images: {}".format(year[3], wl[3], tot4), fontsize=13)

year = [r'2014', r'2016', r'2018', r'2020']
wl = [r'5577 Å', r'5577 Å', r'5577 Å', r'5577 Å']

# Pie chart for each year
#dist_pie_chart(container_Full, LABELS, year, wl, month=False)
#dist_pie_chart(container_Full, LABELS, year, wl, a_less=True, month=False)
#plt.show()

# Pie chart for months
#dist_pie_chart(container_Full, LABELS, year, wl, month=True)
#dist_pie_chart(container_Full, LABELS, year, wl, month=True, a_less=False)
#plt.show()


# Bar charts
bar_chart(container_Full, LABELS, year, wl, a_less=True, month=False)
bar_chart(container_Full, LABELS, year, wl, a_less=False, month=False)

plt.show()
exit()


# See which class combinations the model had a hard time predicting
def pred_5050(container, title=''):

    #weight = []
    dict = {}
    count_5050 = 0
    count_less_than_60 = 0
    count_over_85 = 0
    index = 0

    for entry in container:

        #print(entry.score[entry.label])
        #print(entry.score)
        index += 1

        if entry.score[entry.label] < 0.6:  # Less than 60 %

            #print("Max: %s " %entry.label, entry.score[entry.label])
            #print('Check second highest label')
            #weight.append(1)

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
            #weight.append(2)
            dict[index] = None
            #print("Max: %.5f [%s]" %(entry.score[entry.label], entry.label))
            if entry.score[entry.label] > 0.85:
                count_over_85 += 1


    #print(weight)
    print(title)
    #print(len(dict))
    print("Over 85% pred acc.: {} [{:.2f}%]".format(count_over_85, (count_over_85/len(dict))*100))
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


def weights_and_stuff():
    print('weights and stuff')

    pred_5050(container_Full, title='Green. 2014, 2016, 2018 and 2020')
    """
    Over 80% pred acc.: 226792 [51.33%]
    50/50 labels: 22484 [5.09%]
    (Less than 60% accuracy, but not 50/50: 39692)
    count: 2410, combi: ('aurora-less', 'arc') (10.7%)
    count: 5597, combi: ('aurora-less', 'diffuse') (24.9%)
    count: 505, combi: ('aurora-less', 'discrete') (2.2%)
    count: 2931, combi: ('arc', 'diffuse') (13.0%)
    count: 2110, combi: ('arc', 'discrete') (9.4%)
    count: 8931, combi: ('diffuse', 'discrete') (39.7%)
    """

#weights_and_stuff()

def max_min_mean(list, label):

    print("{}, max, min, mean:".format(label))
    print(np.max(list))
    print(np.min(list))
    print(np.mean(list))

def neg_pos(list_, label):

    neg_count = len(list(filter(lambda x: (float(x) < 0), list_)))
    pos_count = len(list(filter(lambda x: (float(x) >= 0), list_)))
    neg_per = (neg_count/len(list_))*100
    pos_per = (pos_count/len(list_))*100
    print("\nNr. of entries [{}] with neg and pos Bz values".format(label))
    print("neg: %g [%3.2f%%]" %(neg_count, neg_per))
    print("pos: %g [%3.2f%%]" %(pos_count, pos_per))
    #return neg_count, pos_count

    return neg_per, pos_per


def omni_ting(container, year_='2014', year=False):

    # List to add Bz value
    a_less = []
    arc = []
    diff = []
    disc = []

    count99 = 0
    count99_aless = 0
    input = 'Bz, nT (GSM)'

    if year:
        for entry in container:
            if entry.timepoint[:4] == year_:
                if entry.label == LABELS[0]:
                    if float(entry.solarwind[input]) != 9999.99:
                        a_less.append(float(entry.solarwind[input]))
                    else:
                        count99_aless += 1
                elif entry.label == LABELS[1]:
                    if float(entry.solarwind[input]) != 9999.99:
                        arc.append(float(entry.solarwind[input]))
                    else:
                        count99 += 1
                elif entry.label == LABELS[2]:
                    if float(entry.solarwind[input]) != 9999.99:
                        diff.append(float(entry.solarwind[input]))
                    else:
                        count99 += 1
                elif entry.label == LABELS[3]:
                    if float(entry.solarwind[input]) != 9999.99:
                        disc.append(float(entry.solarwind[input]))
                    else:
                        count99 += 1

    else:
        for entry in container:
            if entry.label == LABELS[0]:
                if float(entry.solarwind[input]) != 9999.99:
                    a_less.append(float(entry.solarwind[input]))
                else:
                    count99_aless += 1
            elif entry.label == LABELS[1]:
                if float(entry.solarwind[input]) != 9999.99:
                    arc.append(float(entry.solarwind[input]))
                else:
                    count99 += 1
            elif entry.label == LABELS[2]:
                if float(entry.solarwind[input]) != 9999.99:
                    diff.append(float(entry.solarwind[input]))
                else:
                    count99 += 1
            elif entry.label == LABELS[3]:
                if float(entry.solarwind[input]) != 9999.99:
                    disc.append(float(entry.solarwind[input]))
                else:
                    count99 += 1

    '''
    max_min_mean(list=a_less, label=LABELS[0])
    max_min_mean(list=arc, label=LABELS[1])
    max_min_mean(list=diff, label=LABELS[2])
    max_min_mean(list=disc, label=LABELS[3])
    '''

    neg_per_less, pos_per_less = neg_pos(a_less, LABELS[0])
    neg_per_arc, pos_per_arc = neg_pos(arc, LABELS[1])
    neg_per_diff, pos_per_diff = neg_pos(diff, LABELS[2])
    neg_per_disc, pos_per_disc = neg_pos(disc, LABELS[3])

    neg = [neg_per_less, neg_per_arc, neg_per_diff, neg_per_disc]
    pos = [pos_per_less, pos_per_arc, pos_per_diff, pos_per_disc]

    '''
    #print("Nr. of [{}] entries: {}".format(LABELS[0], len(a_less)))
    print("Nr. of [{}] entries: {}".format(LABELS[1], len(arc)))
    print("Nr. of [{}] entries: {}".format(LABELS[2], len(diff)))
    print("Nr. of [{}] entries: {}".format(LABELS[3], len(disc)))
    '''
    print("Nr of entries (aurora) with 9999.99 value:    ", count99)
    print("Nr of entries (no aurora) with 9999.99 value: ", count99_aless)


    return a_less, arc, diff, disc, neg, pos

def sub_plots_Bz(year, a_less, arc, diff, disc, neg, pos, T_Aurora_N=None, month_name=None,  N=4):

    bins = np.linspace(-20, 20, 41)

    if year[:4] == '2020':
        shape = '*-'
    elif year[:4] == '2014':
        shape = '.-'
    elif year[:4] == '2016':
        shape = 'o-'
    else:
        shape = 'x-'

    if month_name != None:
        plt.suptitle(r'$B_z$ distribution for all classes {} {}'.format(month_name, year[:4]), fontsize=26) # 18
    else:
        plt.suptitle(r'$B_z$ distribution for all classes. {}'.format(year[:4]), fontsize=26)

    #subplot(N,1,1)
    subplot(N/2,N/2,1)
    plt.title(r'arc', fontsize = 22)
    a_heights, a_bins = np.histogram(arc[0], bins=bins, density=True)
    b_heights, b_bins = np.histogram(arc[1], bins=bins, density=True)
    plt.plot(a_bins[:-1], a_heights, 'o-', label=r'dayside')
    plt.plot(b_bins[:-1], b_heights, '*-', label=r'nightside')
    plt.text(-19, 0.26, r'$B_z < 0$:  {:.1f}%'.format(neg[0][1]), fontsize = 19, color='C0')
    plt.text(4, 0.26, r'$B_z >= 0$: {:.1f}%'.format(pos[0][1]), fontsize = 19, color='C0')
    plt.text(-19, 0.21, r'$B_z < 0$:  {:.1f}%'.format(neg[1][1]), fontsize = 19, color='C1')
    plt.text(4, 0.21, r'$B_z >= 0$: {:.1f}%'.format(pos[1][1]), fontsize = 19, color='C1')

    #plt.text(-19, 0.1, 'Max: {:.1f}, Min: {:.1f}'.format(np.max(pos[0][1]), np.min(neg[0][1])), fontsize = 17, color='C0')
    #plt.text(-19, 0.13, 'Max: {:.1f}, Min: {:.1f}'.format(np.max(pos[1][1]), np.min(neg[1][1])), fontsize = 17, color='C1')
    plt.axvline(x=0, ls='--', color='lightgrey')
    #plt.plot(hours, T_arc_N, shape, label='arc - '+year)
    plt.ylabel("%", fontsize=22, color='r')    # 15
    plt.ylim(-0.01, 0.30)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19) # 11
    plt.legend(fontsize=20, loc='upper left', bbox_to_anchor=(0.71, 1.29),
          fancybox=True, shadow=True, ncol=2)   # 13, bbox_to_anchor=(0.675, 1.2)
    #plt.legend(fontsize=13, shadow=True) #bbox_to_anchor = (1.05, 0.95),
    #plot(hours, T_arc_N, 'arc', year, month=None, monthly=False)

    #subplot(N,1,2)
    subplot(N/2,N/2,2)
    plt.title(r'diffuse', fontsize = 22)
    a_heights, a_bins = np.histogram(diff[0], bins=bins, density=True)
    b_heights, b_bins = np.histogram(diff[1], bins=bins, density=True)
    plt.text(-19, 0.26, r'$B_z < 0$:  {:.1f}%'.format(neg[0][2]), fontsize = 19, color='C0')
    plt.text(4, 0.26, r'$B_z >= 0$: {:.1f}%'.format(pos[0][2]), fontsize = 19, color='C0')
    plt.text(-19, 0.21, r'$B_z < 0$:  {:.1f}%'.format(neg[1][2]), fontsize = 19, color='C1')
    plt.text(4, 0.21, r'$B_z >= 0$: {:.1f}%'.format(pos[1][2]), fontsize = 19, color='C1')
    plt.plot(a_bins[:-1], a_heights, 'o-', label=r'dayside')
    plt.plot(b_bins[:-1], b_heights, '*-', label=r'nightside')
    plt.axvline(x=0, ls='--', color='lightgrey')
    #plot(hours, T_diff_N, 'diffuse', year, month=None, monthly=False)
    #plt.plot(hours, T_diff_N, shape, label='diffuse - '+year)
    plt.ylabel("%", fontsize=22, color='r')    # 15
    plt.ylim(-0.01, 0.30)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19) # 11
    #plt.legend(fontsize=13, bbox_to_anchor = (1.05, 0.95), shadow=True)
    #plt.legend(fontsize=13, shadow=True) #bbox_to_anchor = (1.05, 0.95),

    #subplot(N,1,3)
    subplot(N/2,N/2,3)
    plt.title(r'discrete', fontsize = 22)
    a_heights, a_bins = np.histogram(disc[0], bins=bins, density=True)
    b_heights, b_bins = np.histogram(disc[1], bins=bins, density=True)
    plt.text(-19, 0.26, r'$B_z < 0$:  {:.1f}%'.format(neg[0][3]), fontsize = 19, color='C0')
    plt.text(4, 0.26, r'$B_z >= 0$: {:.1f}%'.format(pos[0][3]), fontsize = 19, color='C0')
    plt.text(-19, 0.21, r'$B_z < 0$:  {:.1f}%'.format(neg[1][3]), fontsize = 19, color='C1')
    plt.text(4, 0.21, r'$B_z >= 0$: {:.1f}%'.format(pos[1][3]), fontsize = 19, color='C1')
    plt.plot(a_bins[:-1], a_heights, 'o-', label=r'dayside')
    plt.plot(b_bins[:-1], b_heights, '*-', label=r'nightside')
    plt.axvline(x=0, ls='--', color='lightgrey')
    #plot(hours, T_disc_N, 'discrete', year, month=None, monthly=False)
    #plt.plot(hours, T_disc_N, shape, label='discrete - '+year)
    plt.ylabel(r"count (normalized)", fontsize=22, color='r')    # 15
    plt.ylim(-0.01, 0.30)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19) # 11
    #plt.legend(fontsize=13, shadow=True) #bbox_to_anchor = (1.05, 0.95),
    plt.xlabel(r"$B_z$ [nT] (GSM)", fontsize=22)    # 15
    # r'W1 disk and central $\pm2^\circ$ subtracted'

    #subplot(N,1,4)
    subplot(N/2,N/2,4)
    plt.title(r'no aurora', fontsize = 22)   # 15
    a_heights, a_bins = np.histogram(a_less[0], bins=bins, density=True)
    b_heights, b_bins = np.histogram(a_less[1], bins=bins, density=True)
    plt.text(-19, 0.26, r'$B_z < 0$:  {:.1f}%'.format(neg[0][0]), fontsize = 19, color='C0')    # 13
    plt.text(4, 0.26, r'$B_z >= 0$: {:.1f}%'.format(pos[0][0]), fontsize = 19, color='C0')
    plt.text(-19, 0.21, r'$B_z < 0$:  {:.1f}%'.format(neg[1][0]), fontsize = 19, color='C1')
    plt.text(4, 0.21, r'$B_z >= 0$: {:.1f}%'.format(pos[1][0]), fontsize = 19, color='C1')
    plt.plot(a_bins[:-1], a_heights, 'o-', label='dayside')
    plt.plot(b_bins[:-1], b_heights, '*-', label='nightside')
    plt.axvline(x=0, ls='--', color='lightgrey')
    #plot(hours, T_c_N, 'no aurora', year, month=None, monthly=False, axis=True)
    #plt.plot(hours, T_c_N, shape, label='no aurora - '+year)
    #plt.xlabel("Hour of the day", fontsize=13)
    plt.ylabel(r"percentage", fontsize=22, color='r')    # 15
    plt.ylim(-0.01, 0.30)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19) # 11
    #plt.legend(fontsize=13, shadow=True) #bbox_to_anchor = (1.05, 0.95),

    plt.xlabel(r"$B_z$ [nT] (GSM)", fontsize=22)    # 15

    plt.subplots_adjust(top=0.84)



def Bz_stats(year):
    print('Bz stats')
    print(year)
    #plt.figure(figsize=(8, 11)) # bredde, hoyde
    plt.figure(figsize=(18, 9)) # bredde, hoyde. 11, 8

    if year == 'All years':
        a_less_Day, arc_Day, diff_Day, disc_Day, neg_Day, pos_Day = omni_ting(container_D)
        a_less_Night, arc_Night, diff_Night, disc_Night, neg_Night, pos_Night = omni_ting(container_N)
    else:
        a_less_Day, arc_Day, diff_Day, disc_Day, neg_Day, pos_Day = omni_ting(container_D, year, True)
        a_less_Night, arc_Night, diff_Night, disc_Night, neg_Night, pos_Night = omni_ting(container_N, year, True)

    a_less = [a_less_Day, a_less_Night]
    arc = [arc_Day, arc_Night]
    diff = [diff_Day, diff_Night]
    disc = [disc_Day, disc_Night]
    neg = [neg_Day, neg_Night]
    pos = [pos_Day, pos_Night]

    sub_plots_Bz(year, a_less, arc, diff, disc, neg, pos)

    plt.savefig("stats/Green/b2/yearly_Bz_plot_{}_small.png".format(year), bbox_inches="tight")
    #plt.show()


Bz_stats(year='2014')
Bz_stats(year='2016')
Bz_stats(year='2018')
Bz_stats(year='2020')
Bz_stats(year='All years')
exit()


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

def stats_aurora(container, label, year=False, weight=False, Bz=False):
    """
    blabla
    """
    Years = []
    Month = []
    Clock = []
    Hours = []
    TH = []
    max_score = []

    def data(entry, Years, Month, Clock, Hours, TH, max_score, W=1, weight=False):
        Years.append(entry.timepoint[:4])
        #Month.append(entry.timepoint[5:7])
        #Clock.append(entry.timepoint[-8:])
        Hours.append(entry.timepoint[-8:-6])
        #TH.append(entry.timepoint[5:7]+entry.timepoint[-8:-6]) # Month, day and hour
        #max_score.append(entry.score[entry.label])

        if weight:
            for i in range(W):
                Hours.append(entry.timepoint[-8:-6])


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

                    if entry.score[entry.label] < 0.5:
                        # No weight
                        W = 1
                        W = int(entry.score[entry.label]*100)
                        if label == LABELS[0] or LABELS[1] or LABELS[2] or LABELS[3]:
                            if entry.label == label:
                                data(entry, Years, Month, Clock, Hours, TH, max_score, W, weight=True)

                        if label == "aurora": # arc, diffuse and discrete aurora is counted as one
                            if entry.label != LABELS[0]:
                                #print(entry.timepoint) # YYYY-MM-DD hh:mm:ss
                                data(entry, Years, Month, Clock, Hours, TH, max_score, W, weight=True)

                    elif entry.score[entry.label] > 0.8:
                        # weight x 3
                        W = 5
                        W = int(entry.score[entry.label]*100)
                        if label == LABELS[0] or LABELS[1] or LABELS[2] or LABELS[3]:
                            if entry.label == label:
                                data(entry, Years, Month, Clock, Hours, TH, max_score, W, weight=True)

                        if label == "aurora": # arc, diffuse and discrete aurora is counted as one
                            if entry.label != LABELS[0]:
                                #print(entry.timepoint) # YYYY-MM-DD hh:mm:ss
                                data(entry, Years, Month, Clock, Hours, TH, max_score, W, weight=True)

                    else:
                        # weight x 2
                        W = 3
                        W = int(entry.score[entry.label]*100)
                        if label == LABELS[0] or LABELS[1] or LABELS[2] or LABELS[3]:
                            if entry.label == label:
                                data(entry, Years, Month, Clock, Hours, TH, max_score, W, weight=True)

                        if label == "aurora": # arc, diffuse and discrete aurora is counted as one
                            if entry.label != LABELS[0]:
                                #print(entry.timepoint) # YYYY-MM-DD hh:mm:ss
                                data(entry, Years, Month, Clock, Hours, TH, max_score, W, weight=True)

                else:
                    if label == LABELS[0] or LABELS[1] or LABELS[2] or LABELS[3]:
                        if entry.label == label:
                            data(entry, Years, Month, Clock, Hours, TH, max_score)

                    if label == "aurora": # arc, diffuse and discrete aurora is counted as one
                        if entry.label != LABELS[0]:
                            #print(entry.timepoint) # YYYY-MM-DD hh:mm:ss
                            data(entry, Years, Month, Clock, Hours, TH, max_score)

    else:
        for entry in container:

            if weight:

                if entry.score[entry.label] < 0.5:
                    # No weight
                    W = 1
                    W = int(entry.score[entry.label]*100)
                    if label == LABELS[0] or LABELS[1] or LABELS[2] or LABELS[3]:
                        if entry.label == label:
                            data(entry, Years, Month, Clock, Hours, TH, max_score, W, weight=True)

                    if label == "aurora": # arc, diffuse and discrete aurora is counted as one
                        if entry.label != LABELS[0]:
                            #print(entry.timepoint) # YYYY-MM-DD hh:mm:ss
                            data(entry, Years, Month, Clock, Hours, TH, max_score, W, weight=True)

                elif entry.score[entry.label] > 0.8:
                    # weight x 3
                    W = 3
                    W = int(entry.score[entry.label]*100)
                    if label == LABELS[0] or LABELS[1] or LABELS[2] or LABELS[3]:
                        if entry.label == label:
                            data(entry, Years, Month, Clock, Hours, TH, max_score, W, weight=True)

                    if label == "aurora": # arc, diffuse and discrete aurora is counted as one
                        if entry.label != LABELS[0]:
                            #print(entry.timepoint) # YYYY-MM-DD hh:mm:ss
                            data(entry, Years, Month, Clock, Hours, TH, max_score, W, weight=True)

                else:
                    # weight x 2
                    W = 2
                    W = int(entry.score[entry.label]*100)
                    if label == LABELS[0] or LABELS[1] or LABELS[2] or LABELS[3]:
                        if entry.label == label:
                            data(entry, Years, Month, Clock, Hours, TH, max_score, W, weight=True)

                    if label == "aurora": # arc, diffuse and discrete aurora is counted as one
                        if entry.label != LABELS[0]:
                            #print(entry.timepoint) # YYYY-MM-DD hh:mm:ss
                            data(entry, Years, Month, Clock, Hours, TH, max_score, W, weight=True)

            else:
                if label == LABELS[0] or LABELS[1] or LABELS[2] or LABELS[3]:
                    if entry.label == label:
                        data(entry, Years, Month, Clock, Hours, TH, max_score)

                if label == "aurora": # arc, diffuse and discrete aurora is counted as one
                    if entry.label != LABELS[0]:
                        #print(entry.timepoint) # YYYY-MM-DD hh:mm:ss
                        data(entry, Years, Month, Clock, Hours, TH, max_score)


    return Years, Month, Clock, Hours, TH, max_score

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


    # OBS!! normalisering her?
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


def sub_plots(year, hours, T_c_N, T_arc_N, T_diff_N, T_disc_N, T_Aurora_N=None, month_name=None,  N=4):

    #f = plt.figure(figsize=(10,3))

    if year[:4] == '2020':
        shape = '*-'
    elif year[:4] == '2014':
        shape = '.-'
    elif year[:4] == '2016':
        shape = 'o-'
    else:
        shape = 'x-'

    subplot(N,1,1)
    if month_name != None:
        plt.title('Statistics ({}) for all classes'.format(month_name), fontsize=18)
    else:
        if len(year) > 4:
            plt.title('Yearly statistics for all classes (weighted)', fontsize=18)
        else:
            plt.title('Yearly statistics for all classes', fontsize=18)
    plt.plot(hours, T_arc_N, shape, label='arc - '+year)
    plt.ylabel("%", fontsize=15)
    plt.legend(fontsize=13, bbox_to_anchor = (1.05, 0.95), shadow=True) #, ncol=2
    #plot(hours, T_arc_N, 'arc', year, month=None, monthly=False)

    subplot(N,1,2)
    #plot(hours, T_diff_N, 'diffuse', year, month=None, monthly=False)
    plt.plot(hours, T_diff_N, shape, label='diffuse - '+year)
    plt.ylabel("%", fontsize=15)
    plt.legend(fontsize=13, bbox_to_anchor = (1.05, 0.95), shadow=True)

    subplot(N,1,3)
    #plot(hours, T_disc_N, 'discrete', year, month=None, monthly=False)
    plt.plot(hours, T_disc_N, shape, label='discrete - '+year)
    plt.ylabel("%", fontsize=15)
    plt.legend(fontsize=13, bbox_to_anchor = (1.05, 0.95), shadow=True)

    subplot(N,1,4)
    #plot(hours, T_c_N, 'no aurora', year, month=None, monthly=False, axis=True)
    plt.plot(hours, T_c_N, shape, label='no aurora - '+year)
    #plt.xlabel("Hour of the day", fontsize=13)
    plt.ylabel("%", fontsize=15)
    plt.legend(fontsize=13, bbox_to_anchor = (1.05, 0.95), shadow=True)

    if N == 5:
        subplot(N,1,5)
        #plot(hours, T_c_N, 'no aurora', year, month=None, monthly=False, axis=True)
        plt.plot(hours, T_Aurora_N, shape, label='aurora - '+year)
        plt.xlabel("Hour of the day", fontsize=15)
        plt.ylabel("%", fontsize=15)
        plt.legend(fontsize=13, bbox_to_anchor = (1.05, 0.95), shadow=True)
    else:
        plt.xlabel("Hour of the day", fontsize=15)

    #plt.tight_layout(rect=[0,0,0.75,1])


def Hour_subplot(container, year, month_name='Jan', N=4, month=False, weight=False):

    hours = Get_hours()

    Years, Months, Clock, Hours, TH, max_score = stats_aurora(container=container, label="aurora-less", year=year[:4], weight=weight)
    Years_arc, Months_arc, Clock_arc, Hours_arc, TH_arc, max_score_arc = stats_aurora(container=container, label="arc", year=year[:4], weight=weight)
    Years_diff, Months_diff, Clock_diff, Hours_diff, TH_diff, max_score_diff = stats_aurora(container=container, label="diffuse", year=year[:4], weight=weight)
    Years_disc, Months_disc, Clock_disc, Hours_disc, TH_disc, max_score_disc = stats_aurora(container=container, label="discrete", year=year[:4], weight=weight)
    Years_A, Months_A, Clock_A, Hours_A, TH_A, max_score_A = stats_aurora(container=container, label="aurora", year=year[:4], weight=weight)

    if month:
        T_c = []; T_arc = []; T_diff = []; T_disc = []
        T_c_N = []; T_arc_N = []; T_diff_N = []; T_disc_N = []
        T_A = []; T_A_N = []

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
        # Aurora
        TH_01_A, TH_02_A, TH_03_A, TH_10_A, TH_11_A, TH_12_A, \
        TH_01_A_N, TH_02_A_N, TH_10_A_N, TH_11_A_N, TH_12_A_N \
        = get_hour_count_per_month(TH_A, hours)

        T_c.extend([TH_01, TH_02, TH_03, TH_10, TH_11, TH_12])
        T_arc.extend([TH_01_arc, TH_02_arc, TH_03_arc, TH_10_arc, TH_11_arc, TH_12_arc])
        T_diff.extend([TH_01_diff, TH_02_diff, TH_03_diff, TH_10_diff, TH_11_diff, TH_12_diff])
        T_disc.extend([TH_01_disc, TH_02_disc, TH_03_disc, TH_10_disc, TH_11_disc, TH_12_disc])
        T_A.extend([TH_01_A, TH_02_A, TH_03_A, TH_10_A, TH_11_A, TH_12_A])

        # Normalized
        T_c_N.extend([TH_01_N, TH_02_N, TH_10_N, TH_11_N, TH_12_N])
        T_arc_N.extend([TH_01_arc_N, TH_02_arc_N, TH_10_arc_N, TH_11_arc_N, TH_12_arc_N])
        T_diff_N.extend([TH_01_diff_N, TH_02_diff_N, TH_10_diff_N, TH_11_diff_N, TH_12_diff_N])
        T_disc_N.extend([TH_01_disc_N, TH_02_disc_N, TH_10_disc_N, TH_11_disc_N, TH_12_disc_N])
        T_A_N.extend([TH_01_A_N, TH_02_A_N, TH_10_A_N, TH_11_A_N, TH_12_A_N])

        M_ = ['01', '02', '03', '10', '11', '12']
        M_label = ['Jan', 'Feb', 'Mar', 'Oct', 'Nov', 'Dec']
        M_label_N = ['Jan', 'Feb', 'Oct', 'Nov', 'Dec']

        index = M_label_N.index(month_name)
        print(M_label_N[index])

        sub_plots(year, hours, T_c_N[index], T_arc_N[index], T_diff_N[index], T_disc_N[index], T_A_N[index], month_name=month_name, N=N)


        '''
        for i in range(len(M_label_N)):
            #plot_hourly_nor(hours, T_c_N[i], T_arc_N[i], T_diff_N[i], T_disc_N[i], year, M_label_N[i], monthly=True)
            sub_plots(year, hours, T_c_N[i], T_arc_N[i], T_diff_N[i], T_disc_N[i], T_Aurora_N=None,  N=4)
        '''

        M_plots = False

        if M_plots:
            for i in range(len(M_)):

                plt.figure()
                subcategorybar(hours, [T_arc[i], T_diff[i], T_disc[i]], ["arc. tot: %s" %sum(T_arc[i]), "diff. tot: %s"%sum(T_diff[i]), "disc. tot: %s"%sum(T_disc[i])])
                plt.title("Stats %s, %s" %(M_label[i], year))
                #plt.xticks(rotation='vertical')
                plt.xlabel("Hour of the day"); plt.ylabel("Count")
                #plt.savefig("stats/Green/b2/hour_plot_%s_%s.png" %(year, M_label[i]))


    else:
        # Year
        T_c = []; T_arc = []; T_diff = []; T_disc = []
        T_c_N = []; T_arc_N = []; T_diff_N = []; T_disc_N = []

        for i in range(len(hours)):
            T_c.append(Hours.count(hours[i]))
            T_arc.append(Hours_arc.count(hours[i]))
            T_diff.append(Hours_diff.count(hours[i]))
            T_disc.append(Hours_disc.count(hours[i]))

        T_Aurora = []
        T_Aurora_N  = []
        T_Aurora = [a + b + c for a, b, c in zip(T_arc, T_diff, T_disc)]

        tot_sum = sum(T_c+T_arc+T_diff+T_disc)
        tot_sum_a = sum(T_arc+T_diff+T_disc)

        #print(tot_sum)
        #print("aurora: ", tot_sum_a, "aurora-less: ", sum(T_c))

        use_tot_sum = True
        if use_tot_sum:

            for i in range(len(hours)):
                T_c_N.append((T_c[i]/tot_sum)*100)
                T_arc_N.append((T_arc[i]/tot_sum)*100)
                T_diff_N.append((T_diff[i]/tot_sum)*100)
                T_disc_N.append((T_disc[i]/tot_sum)*100)
                T_Aurora_N.append((T_Aurora[i]/tot_sum)*100)

        else:
            for i in range(len(hours)):
                T_c_N.append((T_c[i]/sum(T_c))*100)
                T_arc_N.append((T_arc[i]/sum(T_arc))*100)
                T_diff_N.append((T_diff[i]/sum(T_diff))*100)
                T_disc_N.append((T_disc[i]/sum(T_disc))*100)
                T_Aurora_N.append((T_Aurora[i]/tot_sum_a)*100)



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

        #plot(hours, T_Aurora_N, 'Aurora', year, month=None, monthly=False)

        #sub_plots(year, hours, T_c, T_arc, T_diff, T_disc, T_Aurora, N=5)



        sub_plots(year, hours, T_c_N, T_arc_N, T_diff_N, T_disc_N, T_Aurora_N, N=N)

        """
        #plot(hours, T_arc_N, 'arc', year, month=None, monthly=False)
        #plot(hours, T_diff_N, 'diffuse', year, month=None, monthly=False)
        #plot(hours, T_disc_N, 'discrete', year, month=None, monthly=False)
        #plot_hourly_nor(hours, T_c_N, T_arc_N, T_diff_N, T_disc_N, year)
        if year == '2014':
            n = 1
            subplot(2,1,n)
            plt.title('Yearly statistics for aurora/no aurora', fontsize=16)
        else:
            n = 2
            subplot(2,1,n)
            plt.xlabel("Hour of the day", fontsize=13)

        plt.plot(hours, T_c_N, '-', label='no aurora - '+year)
        #plt.ylabel("%", fontsize=13)
        plt.plot(hours, T_Aurora_N, '--', label='aurora - '+year)
        plt.ylabel("%", fontsize=13)
        plt.legend(fontsize=11)

        """

#plt.figure(figsize=(8, 11)) # bredde, hoyde
#Hour_subplot(container=container_Full, year="2014", N=5, month=False)
#Hour_subplot(container=container_Full, year="2016", N=5, month=False)
#Hour_subplot(container=container_Full, year="2018", N=5, month=False)
#Hour_subplot(container=container_Full, year="2020", N=5, month=False)

#plt.savefig("stats/Green/b2/yearly_hour_plot.png", bbox_inches="tight")
#plt.show()

plt.figure(figsize=(8, 11)) # bredde, hoyde
Hour_subplot(container=container_Full, year="2014 w", N=5, month=False)
Hour_subplot(container=container_Full, year="2016 w", N=5, month=False)
Hour_subplot(container=container_Full, year="2018 w", N=5, month=False)
Hour_subplot(container=container_Full, year="2020 w", N=5, month=False)

#plt.savefig("stats/Green/b2/yearly_hour_plot_weight.png", bbox_inches="tight")
plt.show()

exit()

MN = ['Jan', 'Nov', 'Dec']

for i in range(len(MN)):

    plt.figure(figsize=(8, 11)) # bredde, hoyde
    Hour_subplot(container=container_Full, year="2014", month_name=MN[i], N=5, month=True)
    Hour_subplot(container=container_Full, year="2016", month_name=MN[i], N=5,month=True)
    Hour_subplot(container=container_Full, year="2018", month_name=MN[i], N=5,month=True)
    Hour_subplot(container=container_Full, year="2020", month_name=MN[i], N=5,month=True)
    #plt.savefig("stats/Green/b2/monthly_hour_plot_leg_{}.png".format(MN[i]), bbox_inches="tight")
    #plt.show()
