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

folder_name = r'C:\Users\Krist\Documents\dataset_subfolders\NEW'
out_path = r'C:\Users\Krist\Documents\dataset_subfolders\NEW_T'

def transform(folder_name, out_path, test=False):

    # Code from Anna/Lasse:
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
transform(folder_name, out_path, test=False)
