import requests
import os
import glob
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from bs4 import BeautifulSoup

from dateutil import rrule
from datetime import datetime


def Times():
    # Make times
    times = []
    for i in range(24):

        if i < 10:
            str = 'ut0%s/' %i
        else:
            str = 'ut%s/' %i
        times.append(str)
    return times


def Dates(start_date='20200101', end_date='20201231'):
    """
    start_date: '20200101'
    end_date:   '20201231'
    """
    dates = []

    for dt in rrule.rrule(rrule.DAILY,
                          dtstart=datetime.strptime(start_date, '%Y%m%d'),
                          until=datetime.strptime(end_date, '%Y%m%d')):

        dates.append(dt.strftime('%Y%m%d'))
    return dates

def download_info(dates, base_url, times):

    total_img_on_website = 0    # Not all images are ASI images

    urls = []
    filenames = []

    #dates = dates[:14]
    for i in range(len(dates)):

        new_base_url = base_url+dates[i][:4]+'/'+dates[i]+'/'
        date_print = "%s-%s-%s" %(dates[i][:4], dates[i][5:7], dates[i][-2:])
        print("*"*20); print("Day: ", date_print); print("*"*20)

        for i in range(len(times)):

            # content of URL
            r = requests.get(new_base_url+times[i])
            new_url = new_base_url+times[i]

            # Parse HTML Code
            soup = BeautifulSoup(r.text, features="lxml")#, features="lxml", 'html.parser'
            # find all images in URL
            images = soup.findAll('a')
            total_img_on_website += len(images)

            for img in images:
                names = img.contents[0]
                fullLink = img.get('href')

                # Only want .png images
                if fullLink.find(".png") != -1:
                    url = new_url+fullLink
                    urls.append(url)
                    name = fullLink
                    filenames.append(name)

            # Find Keogram images that we dont want to download
            remove = []
            for j in range(len(filenames)):
                if "-" in filenames[j]: # Keogram image have 2200-2300 syntax
                    remove.append(filenames[j])
                    print("removes: ", filenames[j])

            # Remove Keogram images from urls and filenames lists
            for k in range(len(remove)):
                urls.remove(new_url+remove[k])
                filenames.remove(remove[k])

            print("New images for %s %s: " %(date_print, times[i][:-1]), len(images))
            print("Total image count: ", len(urls))
            print("----------------------------------------------")

    return urls, filenames, total_img_on_website

# Download images from urls list
def download_images(urls, names, folder_name, total_img_on_website):

    count = 0

    # print total images found in URL
    print(f"Total URLs/images found: {total_img_on_website} ")
    print(f"We want to download {len(urls)} images")
    print("Starting downloading")

    # checking if images is not zero
    if len(urls) != 0:
        for i, image in enumerate(urls):

            try:
                r = requests.get(image).content
                try:
                    # possibility of decode
                    r = str(r, 'utf-8')

                except UnicodeDecodeError:
                    # After checking above condition, Image Download start
                    with open(f"{folder_name}/%s" %names[i], "wb+") as f:
                        f.write(r)

                    # counting number of image downloaded
                    count += 1
            except:
                pass

        if count == total_img_on_website:
            print("All Images Downloaded!")
        else:
            print(f"Total {count} Images Downloaded Out of {total_img_on_website}")

#dates14 = Dates(start_date='20140101', end_date='20141231') # 2014
#dates20 = Dates(start_date='20200101', end_date='20201231') # 2020
dates14 = Dates(start_date='20140101', end_date='20140131')
dates16 = Dates(start_date='20160101', end_date='20160131')
dates18 = Dates(start_date='20180101', end_date='20180131')
dates20 = Dates(start_date='20200101', end_date='20200131')

dates14_dec = Dates(start_date='20141101', end_date='20141231')
dates16_dec = Dates(start_date='20161101', end_date='20161231')
dates18_dec = Dates(start_date='20181101', end_date='20181231')
dates20_dec = Dates(start_date='20201101', end_date='20201231')

times = Times()

station_nya4 = 'nya4' # Ny Aalesund
station_nya6 = 'nya6' # Ny Aalesund

#folder_name= "../DATA_green"
#out_path = "../T_DATA_green"
folder_name_G = "../DATA_4yr_G"
folder_name_R = "../DATA_4yr_R"

out_path_G = "../T_DATA_4yr_G"
out_path_R = "../T_DATA_4yr_R"

start = time.time()
'''
urls20, filenames20, total_img_on_website20 = download_info(dates=dates20, base_url='http://tid.uio.no/plasma/aurora/'+station_nya6+'/5577/', times=times)
download_images(urls=urls20, names=filenames20, folder_name=folder_name, total_img_on_website=total_img_on_website20)

urls14, filenames14, total_img_on_website14 = download_info(dates=dates14, base_url='http://tid.uio.no/plasma/aurora/'+station_nya6+'/5577/', times=times)
download_images(urls=urls14, names=filenames14, folder_name=folder_name, total_img_on_website=total_img_on_website14)

urls14_nya4, filenames14_nya4, total_img_on_website14_nya4 = download_info(dates=dates14, base_url='http://tid.uio.no/plasma/aurora/'+station_nya4+'/5577/', times=times)
download_images(urls=urls14_nya4, names=filenames14_nya4, folder_name=folder_name, total_img_on_website=total_img_on_website14_nya4)
#download_images(urls=urls, names=filenames, folder_name=folder_name)
'''

def DL_green(times, dates, station, folder_name):
    urls, filenames, total_img_on_website = download_info(dates=dates, base_url='http://tid.uio.no/plasma/aurora/'+station+'/5577/', times=times)
    download_images(urls=urls, names=filenames, folder_name=folder_name, total_img_on_website=total_img_on_website)

def DL_red(times, dates, station, folder_name):
    urls, filenames, total_img_on_website = download_info(dates=dates, base_url='http://tid.uio.no/plasma/aurora/'+station+'/6300/', times=times)
    download_images(urls=urls, names=filenames, folder_name=folder_name, total_img_on_website=total_img_on_website)


#DL_green(times, dates14, station_nya6, folder_name_G)
#DL_green(times, dates14_dec, station_nya6, folder_name_G)
# nya station as well for 2014
#DL_green(times, dates14, station_nya4, folder_name_G)
#DL_green(times, dates14_dec, station_nya4, folder_name_G)

DL_green(times, dates16, station_nya6, folder_name_G)
DL_green(times, dates16_dec, station_nya6, folder_name_G)
DL_green(times, dates18, station_nya6, folder_name_G)
DL_green(times, dates18_dec, station_nya6, folder_name_G)
#DL_green(times, dates20, station_nya6, folder_name_G)
#DL_green(times, dates20_dec, station_nya6, folder_name_G)
'''
DL_red(times, dates14, station_nya6, folder_name_R)
DL_red(times, dates14_dec, station_nya6, folder_name_R)
# nya station as well for 2014
DL_red(times, dates14, station_nya4, folder_name_R)
DL_red(times, dates14_dec, station_nya4, folder_name_R)

DL_red(times, dates16, station_nya6, folder_name_R)
DL_red(times, dates16_dec, station_nya6, folder_name_R)
DL_red(times, dates18, station_nya6, folder_name_R)
DL_red(times, dates18_dec, station_nya6, folder_name_R)
DL_red(times, dates20, station_nya6, folder_name_R)
DL_red(times, dates20_dec, station_nya6, folder_name_R)

'''

stop = time.time() - start
print("Download time for all images [h]: ", stop/(60*60))

def transform(folder_name, out_path, test=False):

    # Code from Anna/Lasse:
    print("Transform downloaded images")
    start_t = time.time()

    for imfile in glob.iglob(folder_name+'/*.png'):

        # Get image filename (tail):
        head, tail = os.path.split(imfile)

        # Read image:
        img = io.imread(imfile, as_gray=True)

        # Scaling image:
        nimg = img/np.percentile(img, 99)
        nimg[nimg > 1] = 1
        nimg = (nimg*(2**16-1)).astype(np.uint16)

        # Save the scaled image:
        io.imsave(out_path+'/'+f'{tail}', nimg)

    stop_t = time.time() - start_t
    print("Image transform time [h]: ", stop_t/(60*60))

    # Test image
    if test:
        for imfile2 in glob.iglob(out_path+'/*.png'):

            head, tail = os.path.split(imfile2)
            img = io.imread(imfile2)
            io.imsave('t_test.png', img)
            plt.imshow(img)
            plt.show()
            exit()

# Transform full image folder
#transform(folder_name_R, out_path_R)
transform(folder_name_G, out_path_G, test=True)
