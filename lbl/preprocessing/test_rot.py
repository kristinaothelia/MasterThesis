from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd
import sys


def rotate_scipy(input, angle):

    #nd.rotate(input, angle, axes=1, 0, reshape=True, output=None, order=3, mode='constant', cval=0.0, prefilter=True)
    return nd.rotate(input, angle)


def crop_image(input):

    # Open the input image as numpy array, convert to RGB
    img = input.convert("RGB")
    npImage=np.array(img)
    h,w=img.size

    # Create same size mask layer with circle
    mask = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask)
    shape = [0, 0, h, w]  # Need the original shape of image before padded
    draw.ellipse(shape, fill=255)   # Fill white inside circle

    # Convert mask Image to numpy array
    npMask=np.array(mask)
    # Add mask layer to RGB
    npImage=np.dstack((npImage, npMask))

    return Image.fromarray(npImage), mask

def merge_images(original, new, mask, grey_overlap=False):

    # https://note.nkmk.me/en/python-pillow-paste/

    if grey_overlap:
        original.paste(new.convert("L"), (0, 0), mask)
    else:
        original.paste(new, (0, 0), mask)

    return original




# FUNKER IKKE, men mulig noe av dette maa implementeres for riktige vinkler osv..
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

    # Import image
    img = Image.open('nime1_crop.png')
    #img = Image.open("bilde1.png")     # Feil input med aurora bilde og PIL

    def Rotate_then_crop(img, angle):

        bg_img = img.copy()

        # Rotate
        rot_np = rotate_scipy(np.asarray(img), angle)
        rot_img = Image.fromarray(np.uint8(rot_np))
        #rot_img.show()

        # Crop and make mask
        crop_img, Mask_im = crop_image(rot_img)
        #Mask_im.show()
        #crop_img.show()

        new_img = merge_images(bg_img, rot_img, Mask_im)
        new_img.show()
        # Convert new image to numpy array
        new_img_np = np.asarray(bg_img)
        new_img_np_grey = np.asarray(bg_img.convert("L"))

        print("Array in RGB:    ",new_img_np.shape)
        print("Array in grey/L: ", new_img_np_grey.shape)

        # return new_img_np

    # Funker bare bra for 90, 180, 270, pga roterer firkantet bilde..
    Rotate_then_crop(img,  90)
    Rotate_then_crop(img, 75)
    #Rotate_then_crop(img, 270)

    def Crop_then_rotate(img, angle):

        bg_img = img.copy()
        print("original image: ", bg_img.size)

        # Crop image
        crop_img, Mask_im = crop_image(img)
        #Mask_im.show()
        crop_img.show()
        print("cropped image: ", crop_img.size)

        # Rotate image
        rot_np = rotate_scipy(np.asarray(crop_img), angle)
        #rot_np = rotate_scipy(np.asarray(img), angle)
        rot_img = Image.fromarray(np.uint8(rot_np))
        rot_img.show()
        print("rotated image: ", rot_img.size)

        # Merge images. NOT WORKING
        #bg_img.paste(img.convert('L'), (0, 0), Mask_im)
        bg_img.paste(rot_img, (0, 0), Mask_im)
        #bg_img.paste(rot_img, (0,20))   # Senker bilde med 20, viser hvit kant..
        bg_img.show()

    # Den beste metoden? Men faar ikke merget bildene paa slutten, pga;
    # self.im.paste(im, box, mask.im)
    # ValueError: images do not match
    # Hvit kant rundt rotated/cropped image

    Crop_then_rotate(img, 75)
    #Crop_then_rotate(img, 180)

    '''
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
    '''
