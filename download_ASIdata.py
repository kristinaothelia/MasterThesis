#from bs4 import *
import requests
import os
import glob
import numpy as np

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

        #print(dt.strftime('%Y%m%d'))
        dates.append(dt.strftime('%Y%m%d'))
    return dates

#dates = Dates(start_date='20140101', end_date='20141231') # 2014
dates = Dates(start_date='20200101', end_date='20201231') # 2020
times = Times()
#station = 'nya6' # Ny Aalesund
#wavelength = ['5577', '6300']
#base_url = 'http://tid.uio.no/plasma/aurora/'+station+'/'+wavelength[1]+'/'
base_url = 'http://tid.uio.no/plasma/aurora/nya6/6300/'

urls = []
filenames = []

folder_name= "../TEST"
out_path = "../TESTTEST"
total_img_on_website = 0    # Not all images are ASI images

dates = dates[:14]

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

# Download images from urls list
def download_images(urls, names, folder_name):

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


download_images(urls=urls, names=filenames, folder_name=folder_name)

# Code from Anna/Lasse:
print("Transform downloaded images")
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