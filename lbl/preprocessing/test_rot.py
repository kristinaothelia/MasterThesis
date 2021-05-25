from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt
import numpy as np


# FUNKER IKKE


def circle_rotate(image, x, y, radius, degree):

    box = (x-radius, y-radius, x+radius+1, y+radius+1)
    image = Image.fromarray(np.uint8(image)) # back to image
    crop = image.crop(box=box)
    crop_arr = np.asarray(crop)

    # build the cirle mask
    mask = np.zeros((2*radius+1, 2*radius+1))
    for i in range(crop_arr.shape[0]):
        for j in range(crop_arr.shape[1]):
            if (i-radius)**2 + (j-radius)**2 <= radius**2:
                mask[i,j] = 1

    # create the new circular image
    print(crop_arr.shape)
    sub_img_arr = np.empty(crop_arr.shape ,dtype='uint8')
    #sub_img_arr[:,:,:3] = crop_arr[:,:,:3]
    #sub_img_arr[:,:,3] = mask*255
    sub_img_arr[:,:] = crop_arr[:,:]
    sub_img_arr[:,:] = mask#*255
    #sub_img = Image.fromarray(sub_img_arr, "RGBA").rotate(degree)
    sub_img = Image.fromarray(sub_img_arr, "L").rotate(degree)

    i2 = image.copy()
    i2.show()
    #i2.paste(sub_img, box[:2], sub_img.convert('RGBA'))
    i2.paste(sub_img, box[:2], sub_img.convert('L'))

    return i2


if __name__ == '__main__':


    img = np.zeros(shape=(2, 469, 469))
    img[:,:200] = 1
    plt.imshow(img[0])
    #plt.show()


    img = np.asarray(Image.open('nime1_crop.png'))
    img = np.asarray(Image.open('nime1_crop.png').convert('L')) # 1 channel

    # Wrong with aurora input !
    img = np.asarray(Image.open("bilde1.png"))
    print(img)
    print("image shape: ",img.shape)
    x = int(img.shape[0]/2)
    y = int(img.shape[1]/2)

    i2  = circle_rotate(image=img, x=x, y=y, radius=y, degree=180)
    i2.show()
