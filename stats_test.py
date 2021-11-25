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

LABELS = ['no aurora', 'arc', 'diffuse', 'discrete']

predicted_file_G = r'C:\Users\Krist\Documents\ASI_json_files\Aurora_G_predicted_efficientnet-b2.json'
predicted_file_1618_G = r'C:\Users\Krist\Documents\ASI_json_files\Aurora_1618_G_predicted_efficientnet-b2.json'

container_G = DatasetContainer.from_json(predicted_file_G)
container_1618_G = DatasetContainer.from_json(predicted_file_1618_G)
print("len container G: ", len(container_G))
print("len container R: ", len(container_1618_G))

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

def distribution(container, labels, year=None, month=False):

    n_less = 0; n_arc = 0; n_diff = 0; n_disc = 0; f = 0; tot = 0

    jan = 0;     feb = 0;     mar = 0;     oct = 0;     nov = 0;     dec = 0
    jan_arc = 0;     feb_arc = 0;     mar_arc = 0;     oct_arc = 0;     nov_arc = 0;     dec_arc = 0
    jan_diff = 0;     feb_diff = 0;     mar_diff = 0;     oct_diff = 0;     nov_diff = 0;     dec_diff = 0
    jan_disc = 0;     feb_disc = 0;     mar_disc = 0;     oct_disc = 0;     nov_disc = 0;     dec_disc = 0

    def M(entry, jan, feb, mar, oct, nov, dec):
        if entry.timepoint[5:7] == '01':
            jan += 1
        if entry.timepoint[5:7] == '02':
            feb += 1
        if entry.timepoint[5:7] == '03':
            mar += 1
        if entry.timepoint[5:7] == '10':
            oct += 1
        if entry.timepoint[5:7] == '11':
            nov += 1
        if entry.timepoint[5:7] == '12':
            dec += 1

        return jan, feb, mar, oct, nov, dec

    for entry in container:

        if year:
            if month:
                if entry.timepoint[:4] == year:
                    tot += 1
                    if entry.label == LABELS[1]:
                        n_arc += 1
                        jan_arc, feb_arc, mar_arc, oct_arc, nov_arc, dec_arc \
                        = M(entry, jan_arc, feb_arc, mar_arc, oct_arc, nov_arc, dec_arc)
                    elif entry.label == LABELS[2]:
                        n_diff += 1
                        jan_diff, feb_diff, mar_diff, oct_diff, nov_diff, dec_diff \
                        = M(entry, jan_diff, feb_diff, mar_diff, oct_diff, nov_diff, dec_diff)
                    elif entry.label == LABELS[3]:
                        n_disc += 1
                        jan_disc, feb_disc, mar_disc, oct_disc, nov_disc, dec_disc \
                        = M(entry, jan_disc, feb_disc, mar_disc, oct_disc, nov_disc, dec_disc)
                    else:
                        n_less += 1
                        jan, feb, mar, oct, nov, dec = M(entry, jan, feb, mar, oct, nov, dec)

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
                        jan_arc, feb_arc, mar_arc, oct_arc, nov_arc, dec_arc = M(entry, jan, feb, mar, oct, nov, dec)
                    elif entry.label == LABELS[2]:
                        n_diff += 1
                        jan_diff, feb_diff, mar_diff, oct_diff, nov_diff, dec_diff = M(entry, jan, feb, mar, oct, nov, dec)
                    elif entry.label == LABELS[3]:
                        n_disc += 1
                        jan_disc, feb_disc, mar_disc, oct_disc, nov_disc, dec_disc = M(entry, jan, feb, mar, oct, nov, dec)
                    else:
                        n_less += 1
                        jan, feb, mar, oct, nov, dec = M(entry, jan, feb, mar, oct, nov, dec)

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

    '''
    print("%23s: %g (%3.1f%%)" %('Total classified images', tot, (tot/len(container))*100))
    print("%23s: %4g (%3.1f%%)" %(labels[0], n_less, (n_less/tot)*100))
    print("%23s: %4g (%3.1f%%)" %(labels[1], n_arc, (n_arc/tot)*100))
    print("%23s: %4g (%3.1f%%)" %(labels[2], n_diff, (n_diff/tot)*100))
    print("%23s: %4g (%3.1f%%)" %(labels[3], n_disc, (n_disc/tot)*100))
    print("Nr. of labels other than classes: ", f)
    '''

    print(sum([jan, jan_arc, jan_diff, jan_disc]))
    print(sum([feb, feb_arc, feb_diff, feb_disc]))
    print(sum([dec, dec_arc, dec_diff, dec_disc]))

    n_less_M = []
    n_arc_M = []
    n_diff_M = []
    n_disc_M = []

    n_less_M.extend([jan, feb, mar, oct, nov, dec])
    n_arc_M.extend([jan_arc, feb_arc, mar_arc, oct_arc, nov_arc, dec_arc])
    n_diff_M.extend([jan_diff, feb_diff, mar_diff, oct_diff, nov_diff, dec_diff])
    n_disc_M.extend([jan_disc, feb_disc, mar_disc, oct_disc, nov_disc, dec_disc])

    return tot, n_less, n_arc, n_diff, n_disc, n_less_M, n_arc_M, n_diff_M, n_disc_M

def subpie(sizes1, sizes2, labels, colors, explode, wl1, wl2, yr1, yr2):
    title = "Distribution."
    if len(labels) == 3:
        title = "Aurora dist."

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8, 4.5))
    fig.suptitle(title, fontsize=16)
    ax1.pie(sizes1, explode=explode, labels=labels, colors=colors, autopct = '%1.1f%%',
            shadow=True, textprops={'fontsize': 11})    #, startangle=90
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title("{}, {}".format(yr1, wl1), fontsize=13)

    ax2.pie(sizes2, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, textprops={'fontsize': 11})#, startangle=90
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax2.set_title("{}, {}".format(yr2, wl2), fontsize=13)
    #plt.legend(fontsize=11)
    #plt.show()

def month_subpie(n_less, n_arc, n_diff, n_disc, labels, title=''):

    jan = [n_less[0], n_arc[0], n_diff[0], n_disc[0]]
    feb = [n_less[1], n_arc[1], n_diff[1], n_disc[1]]
    mar = [n_less[2], n_arc[2], n_diff[2], n_disc[2]]
    oct = [n_less[3], n_arc[3], n_diff[3], n_disc[3]]
    nov = [n_less[4], n_arc[4], n_diff[4], n_disc[4]]
    dec = [n_less[5], n_arc[5], n_diff[5], n_disc[5]]

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2,figsize=(7, 8))
    fig.suptitle("Distribution for {}".format(title), fontsize=16)
    angle = 90

    ax1.pie(jan, explode=explode, labels=labels, colors=colors, autopct = '%1.1f%%',
            shadow=True, textprops={'fontsize': 11}, startangle=angle)    #, startangle=90
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title("Jan", fontsize=13)

    ax2.pie(feb, explode=explode, labels=labels, colors=colors, autopct = '%1.1f%%',
            shadow=True, textprops={'fontsize': 11}, startangle=angle)    #, startangle=90
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax2.set_title("Feb", fontsize=13)

    if mar[0] != 0:
        ax3.pie(mar, explode=explode, labels=labels, colors=colors, autopct = '%1.1f%%',
                shadow=True, textprops={'fontsize': 11}, startangle=angle)    #, startangle=90
        ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax3.set_title("Mar", fontsize=13)
    else:
        ax3.axis('off')

    ax4.pie(oct, explode=explode, labels=labels, colors=colors, autopct = '%1.1f%%',
            shadow=True, textprops={'fontsize': 11}, startangle=angle)    #, startangle=90
    ax4.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax4.set_title("Oct", fontsize=13)

    ax5.pie(nov, explode=explode, labels=labels, colors=colors, autopct = '%1.1f%%',
            shadow=True, textprops={'fontsize': 11}, startangle=angle)    #, startangle=90
    ax5.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax5.set_title("Nov", fontsize=13)

    ax6.pie(dec, explode=explode, labels=labels, colors=colors, autopct = '%1.1f%%',
            shadow=True, textprops={'fontsize': 11}, startangle=angle)    #, startangle=90
    ax6.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax6.set_title("Dec", fontsize=13)
    #plt.legend(fontsize=11)

    #plt.show()


def make_dist_and_pie(container1, container2, wl1, wl2, yr1, yr2, month=False):
    #distribution(container, LABELS)

    tot1, n_less1, n_arc1, n_diff1, n_disc1, \
    n_less_M1, n_arc_M1, n_diff_M1, n_disc_M1 \
    = distribution(container1, LABELS, year=yr1, month=True)

    tot2, n_less2, n_arc2, n_diff2, n_disc2, \
    n_less_M2, n_arc_M2, n_diff_M2, n_disc_M2 \
    = distribution(container2, LABELS, year=yr2, month=True)
    #Pie chart, where the slices will be ordered and plotted counter-clockwise:
    sizes1 = [n_less1, n_arc1, n_diff1, n_disc1]
    sizes2 = [n_less2, n_arc2, n_diff2, n_disc2]

    #aurora14 = sum(sizes14[1]+sizes14[2]+sizes14[3])
    explode = (0.05, 0.05, 0.05, 0.05)
    colors = ['dimgrey','dodgerblue','forestgreen', 'mediumslateblue']

    #subpie(sizes1=sizes14, sizes2=sizes20, labels=LABELS, colors=colors, explode=explode)
    subpie(sizes1=sizes1[1:], sizes2=sizes2[1:], labels=LABELS[1:], colors=colors[1:], explode=explode[1:], wl1=wl1, wl2=wl2, yr1=yr1, yr2=yr2)
    subpie(sizes1=sizes1, sizes2=sizes2, labels=LABELS, colors=colors, explode=explode, wl1=wl1, wl2=wl2, yr1=yr1, yr2=yr2)
    #month_subpie(n_less_M14, n_arc_M14, n_diff_M14, n_disc_M14, LABELS, '2014')
    #month_subpie(n_less_M20, n_arc_M20, n_diff_M20, n_disc_M20, LABELS, '2020')
    #plt.show()

'''
make_dist_and_pie(container_R, container_G, wl1='6300 (Red)', wl2='5577 (Green)', yr1='2014', yr2='2014', month=False)
make_dist_and_pie(container_1618_R, container_1618_G, wl1='6300 (Red)', wl2='5577 (Green)', yr1='2016', yr2='2016', month=False)
make_dist_and_pie(container_1618_R, container_1618_G, wl1='6300 (Red)', wl2='5577 (Green)', yr1='2018', yr2='2018', month=False)
make_dist_and_pie(container_R, container_G, wl1='6300 (Red)', wl2='5577 (Green)', yr1='2020', yr2='2020', month=False)
'''

def yr_4(container, labels, year, wl, a_less=False, month=False):

    tot1, n_less1, n_arc1, n_diff1, n_disc1, \
    n_less_M1, n_arc_M1, n_diff_M1, n_disc_M1 \
    = distribution(container[0], labels, year[0], month)

    tot2, n_less2, n_arc2, n_diff2, n_disc2, \
    n_less_M2, n_arc_M2, n_diff_M2, n_disc_M2 \
    = distribution(container[1], labels, year[1], month)

    tot3, n_less3, n_arc3, n_diff3, n_disc3, \
    n_less_M3, n_arc_M3, n_diff_M3, n_disc_M3 \
    = distribution(container[2], labels, year[2], month)

    tot4, n_less4, n_arc4, n_diff4, n_disc4, \
    n_less_M4, n_arc_M4, n_diff_M4, n_disc_M4 \
    = distribution(container[3], labels, year[3], month)

    #Pie chart, where the slices will be ordered and plotted counter-clockwise:
    sizes1 = [n_less1, n_arc1, n_diff1, n_disc1]
    print(n_less1+ n_arc1+ n_diff1+ n_disc1)
    print(n_less1+ n_arc1+ n_diff1+ n_disc1)
    sizes2 = [n_less2, n_arc2, n_diff2, n_disc2]
    sizes3 = [n_less3, n_arc3, n_diff3, n_disc3]
    sizes4 = [n_less4, n_arc4, n_diff4, n_disc4]

    #aurora14 = sum(sizes14[1]+sizes14[2]+sizes14[3])
    explode = (0.05, 0.05, 0.05, 0.05)
    colors = ['dimgrey','dodgerblue','forestgreen', 'mediumslateblue']

    if a_less:
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

year = ['2014', '2016', '2018', '2020']
container = [container_G, container_1618_G, container_1618_G, container_G]
wl = ['5577 Å', '5577 Å', '5577 Å', '5577 Å']
yr_4(container, LABELS, year, wl, month=False)
#plt.show()
yr_4(container, LABELS, year, wl, a_less= True, month=False)
plt.show()

def pie_fancy(sizes, bar_sizes, labels):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
    fig.subplots_adjust(wspace=0)

    labels = labels
    explode = [0, 0.1]
    angle = 180 * sizes[1]/sum(sizes)
    ax1.pie(sizes, autopct='%1.1f%%', startangle=angle,
            labels=labels, explode=explode)

    # bar chart parameters
    xpos = 0
    bottom = 0
    ratios = bar_sizes/sizes[1]
    width = .2

    for j in range(len(ratios)):
        height = ratios[j]
        ax2.bar(xpos, height, width, bottom=bottom)#, color=colors[j]
        ypos = bottom + ax2.patches[j].get_height() / 2
        bottom += height
        ax2.text(xpos, ypos, "%d%%" % (ax2.patches[j].get_height() * 100),
                 ha='center')

    ax2.set_title('Aurora shape')
    ax2.legend(LABELS[1:])
    ax2.axis('off')
    ax2.set_xlim(- 2.5 * width, 2.5 * width)

    # use ConnectionPatch to draw lines between the two plots
    # get the wedge data
    theta1, theta2 = ax1.patches[1].theta1, ax1.patches[1].theta2
    center, r = ax1.patches[1].center, ax1.patches[1].r
    bar_height = sum([item.get_height() for item in ax2.patches])

    # draw top connecting line
    x = r * np.cos(np.pi / 180 * theta2) + center[0]
    y = r * np.sin(np.pi / 180 * theta2) + center[1]
    con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
                          xyB=(x, y), coordsB=ax1.transData)
    con.set_color([0, 0, 0])
    con.set_linewidth(4)
    ax2.add_artist(con)

    # draw bottom connecting line
    x = r * np.cos(np.pi / 180 * theta1) + center[0]
    y = r * np.sin(np.pi / 180 * theta1) + center[1]
    con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
                          xyB=(x, y), coordsB=ax1.transData)
    con.set_color([0, 0, 0])
    ax2.add_artist(con)
    con.set_linewidth(4)

    plt.show()

#pie_fancy(sizes=[sizes14[0], aurora14], bar_sizes=sizes14[1:], labels=[LABELS[0], 'aurora'])
