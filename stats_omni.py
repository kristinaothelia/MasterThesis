#from lbl.class_corrector import ClassCorrector
from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from pylab import *
# -----------------------------------------------------------------------------

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

# Red:      6300        Count:  142 470
# Green:    5577        Count:  284 840

LABELS = ['aurora-less', 'arc', 'diffuse', 'discrete']

# All 4 years, jan+nov+dec
predicted_G_Full = r'C:\Users\Krist\Documents\ASI_json_files\AuroraFull_G_omni_mean_predicted_efficientnet-b2.json'
container_Full = DatasetContainer.from_json(predicted_G_Full)
print("len container Full: ", len(container_Full))

container_D = DatasetContainer.from_json(r'C:\Users\Krist\Documents\ASI_json_files\AuroraFull_G_omni_mean_predicted_efficientnet-b2_daytime.json')
container_N = DatasetContainer.from_json(r'C:\Users\Krist\Documents\ASI_json_files\AuroraFull_G_omni_mean_predicted_efficientnet-b2_nighttime.json')

print('len container day:   ', len(container_D))
print('len container night: ', len(container_N))

#predicted_file_G = r'C:\Users\Krist\Documents\ASI_json_files\Aurora_G_omni_predicted_efficientnet-b2.json'
#predicted_file_G = r'C:\Users\Krist\Documents\ASI_json_files\Aurora_G_omni_predicted_efficientnet-b2_cut.json'
#predicted_file_G = r'C:\Users\Krist\Documents\ASI_json_files\Aurora_G_omni_mean_predicted_efficientnet-b2.json'
#predicted_file_G = r'C:\Users\Krist\Documents\ASI_json_files\Aurora_G_omni_mean_predicted_efficientnet-b2_cut.json'
#predicted_file_R = r'C:\Users\Krist\Documents\ASI_json_files\Aurora_R_omni_predicted_efficientnet-b2.json'
#predicted_file_1618 = r'C:\Users\Krist\Documents\ASI_json_files\Aurora_1618_G_omni_predicted_efficientnet-b2.json'
#predicted_file_1618 = r'C:\Users\Krist\Documents\ASI_json_files\Aurora_1618_G_omni_mean_predicted_efficientnet-b2.json'

#container_G = DatasetContainer.from_json(predicted_file_G)
#container_R = DatasetContainer.from_json(predicted_file_R)
#container_1618 = DatasetContainer.from_json(predicted_file_1618)
#print("len container G: ", len(container_G))
#print("len container 1618: ", len(container_1618))
#print("len container R: ", len(container_R))


def max_min_mean(list, label):

    print("{}, max, min, mean:".format(label))
    print(np.max(list))
    print(np.min(list))
    print(np.mean(list))

def neg_pos(list_, label):

    neg_count = len(list(filter(lambda x: (float(x) < 0), list_)))
    pos_count = len(list(filter(lambda x: (float(x) >= 0), list_)))
    print("\nNr. of entries [{}] with neg and pos Bz values".format(label))
    print("neg: %g [%3.2f%%]" %(neg_count, (neg_count/len(list_))*100))
    print("pos: %g [%3.2f%%]" %(pos_count, (pos_count/len(list_))*100))
    #return neg_count, pos_count


def omni(container, title=None):

    Bz_values_GSM = []
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
                Bz_values_GSM.append(float(entry.solarwind['Bz, nT (GSM)']))
                Bz_labels.append(entry.label)

        else:
            # aurora-less
            if entry.solarwind['Bz, nT (GSM)'] == 9999.99:
                count99_aless += 1
            else:
                Bz_a_less.append(float(entry.solarwind['Bz, nT (GSM)']))

    print(title);print('-----------------------')

    max_min_mean(list=Bz_values_GSM, label='Aurora')
    max_min_mean(list=Bz_a_less, label=LABELS[0])

    neg_pos(list_=Bz_values_GSM, label='Aurora')
    neg_pos(list_=Bz_a_less, label=LABELS[0])

    print("Nr of entries (aurora) with 9999.99 value:    ", count99)
    print("Nr of entries (no aurora) with 9999.99 value: ", count99_aless)


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

    #neg_pos(a_less, LABELS[0])
    neg_pos(arc, LABELS[1])
    neg_pos(diff, LABELS[2])
    neg_pos(disc, LABELS[3])

    '''
    #print("Nr. of [{}] entries: {}".format(LABELS[0], len(a_less)))
    print("Nr. of [{}] entries: {}".format(LABELS[1], len(arc)))
    print("Nr. of [{}] entries: {}".format(LABELS[2], len(diff)))
    print("Nr. of [{}] entries: {}".format(LABELS[3], len(disc)))
    '''
    print("Nr of entries (aurora) with 9999.99 value:    ", count99)
    print("Nr of entries (no aurora) with 9999.99 value: ", count99_aless)


    return a_less, arc, diff, disc

def Bz(date):

    time = []
    label = []
    Bz = []
    Name = []

    input = 'Bz, nT (GSE)'

    for entry in container:
        #if entry.timepoint[:10] == date:
        if entry.timepoint[:13] == date:
            if entry.solarwind[input] != 9999.99:
                time.append(entry.timepoint)
                label.append(entry.label)
                Bz.append(entry.solarwind[input])
                Name.append(entry.image_path[-33:])


    df = pd.DataFrame()
    df['Date'] = pd.to_datetime(time)#time[:10]
    #df['Clock'] = time[12:-1]
    df['Label'] = label
    df['Bz'] = Bz
    df['Name'] = Name
    df.sort_values(by='Date', inplace=True)
    print(df)

    for i in range(len(label)):
        if df['Label'][i] == 'diffuse':
            plt.plot(df['Date'][i], df['Bz'][i], 'r*')
        if df['Label'][i] == 'discret':
            plt.plot(df['Date'][i], df['Bz'][i], 'g*')
        if df['Label'][i] == 'arc':
            plt.plot(df['Date'][i], df['Bz'][i], 'm*')

    plt.title("red: diffuse, green: discrete, magenta: arc")
    plt.plot(df['Date'], df['Bz'], '-')
    #plt.plot(arc_time_clock, arc, '.')
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()


#Bz(date='2014-12-21 07')
#exit()

'''
#omni(container=container_D)
#omni(container=container_N)
omni(container=container_G, title='Green')
omni(container=container_R, title='Red')
exit()
'''
#print('----------')

def Bz_plots(c, c_Night, title):

    fig, ax = plt.subplots()

    #bins = [-15, -10, -5, 0, 5, 10, 15, 20, 25]
    bins = np.linspace(-25, 25, 51)

    a_heights, a_bins = np.histogram(c, bins=bins, density=True)
    b_heights, b_bins = np.histogram(c_Night, bins=bins, density=True)

    plt.plot(a_bins[:-1], a_heights, '.-', label='day')
    plt.plot(b_bins[:-1], b_heights, '*-', label='night')
    plt.axvline(x=0, ls='--', color='lightgrey')
    plt.title(title)
    plt.xlabel('Bz (GSE) value')
    plt.ylabel('%')
    plt.ylim(0, 0.25)
    plt.legend()

#container_D = DatasetContainer.from_json(r'C:\Users\Krist\Documents\ASI_json_files\Aurora_G_omni_mean_predicted_efficientnet-b2_day.json')
#container_N = DatasetContainer.from_json(r'C:\Users\Krist\Documents\ASI_json_files\Aurora_G_omni_mean_predicted_efficientnet-b2_night.json')

#a_less, arc, diff, disc = omni_ting(container_D)
#a_less_Night, arc_Night, diff_Night, disc_Night = omni_ting(container_N)

#Bz_plots(a_less, a_less_Night, 'no aurora')
#Bz_plots(arc, arc_Night, 'arc')
#Bz_plots(diff, diff_Night, 'diffuse')
#Bz_plots(disc, disc_Night, 'discrete')

#plt.show()

#omni_ting(container_D, '2014', True)
#omni_ting(container_N, '2014', True)
a_less, arc, diff, disc = omni_ting(container_Full) #, '2020', True

fig, ax = plt.subplots()

#bins = [-15, -10, -5, 0, 5, 10, 15, 20, 25]
bins = np.linspace(-25, 25, 51)
bins = np.linspace(-20, 20, 41)

a_heights, a_bins = np.histogram(a_less, bins=bins, density=True)
b_heights, b_bins = np.histogram(arc, bins=bins, density=True)
c_heights, c_bins = np.histogram(diff, bins=bins, density=True)
d_heights, d_bins = np.histogram(disc, bins=bins, density=True)

plt.plot(a_bins[:-1], a_heights, '|-', label='no aurora/clouds')
plt.plot(b_bins[:-1], b_heights, '*-', label='arc')
plt.plot(c_bins[:-1], c_heights, '.-', label='diffuse')
plt.plot(d_bins[:-1], d_heights, 'x-', label='discrete')
plt.axvline(x=0, ls='--', color='lightgrey')
plt.title('Jan, Nov and Dec for 2014, 2016, 2018 and 2020')
plt.xlabel('Bz (GSE) value')
plt.ylabel('%')
plt.ylim(0, 0.2)
plt.legend()
plt.show()

'''
Bz_plots(a_less, a_less_Night, 'no aurora')
Bz_plots(arc, arc_Night, 'arc')
Bz_plots(diff, diff_Night, 'diffuse')
Bz_plots(disc, disc_Night, 'discrete')

plt.show()
'''
#Make plots when Bz <0 and Bz > 0.
#For all classes.
#Hour plot
#Year or Month plot

exit()

def distribution(container, labels, year=None):

    n_less = 0; n_arc = 0; n_diff = 0; n_disc = 0; f = 0; tot = 0

    for entry in container:

        if year:
            #if entry.timepoint[:4] == year:
            if entry.timepoint[:7] == year:
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

    return tot

#distribution(container, LABELS)
#distribution(container_1618, LABELS, year='2016')
#distribution(container_1618, LABELS, year='2018')
'''
tot_14_jan = distribution(container_G, LABELS, year='2014-01')
tot_14_feb = distribution(container_G, LABELS, year='2014-02')
tot_14_dec = distribution(container_G, LABELS, year='2014-12')
tot_20_jan = distribution(container_G, LABELS, year='2020-01')
tot_20_feb = distribution(container_G, LABELS, year='2020-02')
tot_20_dec = distribution(container_G, LABELS, year='2020-12')
#distribution(container, LABELS, year='2020')
print(tot_14_jan + tot_14_feb +tot_14_dec+tot_20_jan+tot_20_feb+tot_20_dec)
exit()
'''
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

def stats_aurora(container, label, year=False, weight=False):
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
                    #elif label == "aurora":
                    if label == "aurora":
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

    if year == '2020':
        shape = '--'
    elif year == '2014':
        shape = '.-'
    elif year == '2016':
        shape = ':'
    else:
        shape = '-'

    subplot(N,1,1)
    if month_name != None:
        plt.title('Statistics ({}) for all classes'.format(month_name), fontsize=18)
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


def Hour_subplot(container, year, month_name='Jan', N=4, month=False):

    hours = Get_hours()

    Years, Months, Clock, Hours, TH = stats_aurora(container=container, label="aurora-less", year=year, weight=False)
    Years_arc, Months_arc, Clock_arc, Hours_arc, TH_arc = stats_aurora(container=container, label="arc", year=year, weight=False)
    Years_diff, Months_diff, Clock_diff, Hours_diff, TH_diff = stats_aurora(container=container, label="diffuse", year=year, weight=False)
    Years_disc, Months_disc, Clock_disc, Hours_disc, TH_disc = stats_aurora(container=container, label="discrete", year=year, weight=False)
    Years_A, Months_A, Clock_A, Hours_A, TH_A = stats_aurora(container=container, label="aurora", year=year, weight=False)


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

        #sub_plots(year, hours, T_c_N, T_arc_N, T_diff_N, T_disc_N, T_Aurora_N=None, N=4)



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

plt.figure(figsize=(8, 11)) # bredde, hoyde
Hour_subplot(container=container_G, year="2014", N=5, month=False)
Hour_subplot(container=container_1618, year="2016", N=5,month=False)
Hour_subplot(container=container_1618, year="2018", N=5,month=False)
Hour_subplot(container=container_G, year="2020", N=5,month=False)

plt.savefig("stats/Green/b2/yearly_hour_plot.png", bbox_inches="tight")
exit()

MN = ['Jan', 'Nov', 'Dec']

for i in range(len(MN)):

    plt.figure(figsize=(8, 11)) # bredde, hoyde
    Hour_subplot(container=container_G, year="2014", month_name=MN[i], N=5, month=True)
    Hour_subplot(container=container_1618, year="2016", month_name=MN[i], N=5,month=True)
    Hour_subplot(container=container_1618, year="2018", month_name=MN[i], N=5,month=True)
    Hour_subplot(container=container_G, year="2020", month_name=MN[i], N=5,month=True)
    plt.savefig("stats/Green/b2/monthly_hour_plot_leg_{}.png".format(MN[i]), bbox_inches="tight")
    #plt.show()


'''
Hour_subplot(year="2014", month_name='Jan', month=True)
Hour_subplot(year="2020", month_name='Jan', month=True)
plt.show()

Hour_subplot(year="2014", month_name='Feb', month=True)
Hour_subplot(year="2020", month_name='Feb', month=True)
plt.show()

Hour_subplot(year="2014", month_name='Oct', month=True)
Hour_subplot(year="2020", month_name='Oct', month=True)
plt.show()

Hour_subplot(year="2014", month_name='Nov', month=True)
Hour_subplot(year="2020", month_name='Nov', month=True)
plt.show()

Hour_subplot(year="2014", month_name='Dec', month=True)
Hour_subplot(year="2020", month_name='Dec', month=True)
plt.show()
'''
