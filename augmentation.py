import cv2
from PIL import Image
import random
import numpy as np
import tensorflow as tf
import os
def flip_image_horizontal(filename, directory):
    img = cv2.imread(directory + filename)
    horizontal_img = cv2.flip(img, 0)
    final_path = directory + '\\Augmented\\' + filename[:filename.index('.')] + '_Horizontal_Flip.png'
    cv2.imwrite(final_path,horizontal_img)
def flip_image_vertical(filename, directory):
    img = cv2.imread(directory + filename)
    vertical_img = cv2.flip(img, 1)
    final_path = directory + '\\Augmented\\' + filename[:filename.index('.')] + '_Vertical_Flip.png'
    cv2.imwrite(final_path,vertical_img)
def rotate_image(filename,directory):
    img = Image.open(directory+filename)
    degrees = random.randrange(-9,9)
    if degrees == 0:
        degrees += random.randrange(1,9)
    rotated_img = img.rotate(degrees)
    final_path = directory + '\\Augmented\\' + filename[:filename.index('.')] + '_Rotated_' + str(degrees) +'.png'
    rotated_img.save(final_path)
def zoom_image(filename,directory):
    img = cv2.imread(directory + filename).copy()
    zoom_per = random.randrange(1,10)
    zoom_per = (zoom_per/100)+1
    height, width = img.shape[:2]
    new_height, new_width = int(height * zoom_per), int(width * zoom_per)
    y1, x1 = max(0, new_height-height) //2, max(0, new_width-width) //2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    bbox = (bbox/zoom_per).astype(int)
    y1,x1,y2,x2 = bbox
    zoomed_img = img[y1:y2,x1:x2]
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)
    result_img = cv2.resize(zoomed_img, (resize_width, resize_height))
    result_img = np.pad(result_img, pad_spec, mode='constant')
    final_path = directory + '\\Augmented\\' + filename[:filename.index('.')] + '_Zoomed_' + str(int((zoom_per-1)*100)) +'.png'
    cv2.imwrite(final_path,result_img)
def adjust_img_saturation(filename, directory):
    img = cv2.imread(directory + filename)
    saturation_per = random.randrange(-42,43)
    if saturation_per == 0:
        saturation_per += random.randrange(1,43)
    saturated_img = tf.image.adjust_saturation(img, saturation_per/10.0)
    final_path = directory + '\\Augmented\\' + filename[:filename.index('.')] + '_Saturated_' + str(saturation_per) +'.png'
    tf.keras.utils.save_img(final_path, saturated_img)
def adjust_img_hue(filename, directory):
    img = cv2.imread(directory + filename)
    hue_per = random.randrange(-44,45)
    if hue_per == 0:
        hue_per += random.randrange(1,45)
    hue_img = tf.image.adjust_hue(img, (hue_per/100))
    final_path = directory + '\\Augmented\\' + filename[:filename.index('.')] + '_Hue_' + str(hue_per) +'.png'
    tf.keras.utils.save_img(final_path, hue_img)
def adjust_img_contrast(filename,directory):
    img = cv2.imread(directory + filename)
    contrast_per = random.randrange(-11,12)
    if contrast_per ==  0:
        contrast_per += random.randrange(1,12)
    contrast_img = tf.image.adjust_contrast(img, (contrast_per/100.0 + 1))
    final_path = directory + '\\Augmented\\' + filename[:filename.index('.')] + '_Contrast_' + str(contrast_per) +'.png'
    tf.keras.utils.save_img(final_path, contrast_img)
#@title augment
def augment_data(directory, count_to_match):
    starting_val = len(os.listdir(directory + 'Augmented\\')) + len(os.listdir(directory)) -1
    #Subtract one to account for the Augmented Folder
    for file in os.listdir(directory):
        if os.path.isfile(directory + file):
            aug_method = random.randrange(0,7)
            if aug_method == 0:
                flip_image_horizontal(file, directory)
            elif aug_method == 1:
                flip_image_vertical(file,directory)
            elif aug_method == 2:
                rotate_image(file, directory)
            elif aug_method == 3:
                zoom_image(file, directory)
            elif aug_method == 4:
                adjust_img_saturation(file, directory)
            elif aug_method == 5:
                adjust_img_hue(file,directory)
            elif aug_method == 6:
                adjust_img_contrast(file, directory)
            starting_val += 1
            if starting_val >= count_to_match:
                break
        else:
            print(f"Not a file:{file}")

    #verify counts
    subfolder = directory + "Augmented\\"
    if (len(os.listdir(directory))-1) + len(os.listdir(directory + "Augmented\\")) == count_to_match:
        print(f"Count verified, total of {count_to_match} images with {len(os.listdir(directory))-1} original and {len(os.listdir(subfolder))} augmented.")
    else:
        print(f"Incorrect amount. Meant to have {count_to_match}, found {(len(os.listdir(directory))-1) + len(os.listdir(subfolder))} .")
def convert_images_to_jpg(target_dir,jpg_dir):
    for filename in os.listdir(target_dir):
        if filename[30] == '_':
            try:
                img = cv2.imread(target_dir + filename)
                savestring = jpg_dir + filename[:-3] + "jpg"
                cv2.imwrite(savestring, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            except Exception as e:
                print(e, filename)
        elif filename[30] == '.':
            jpg_file_path = jpg_dir
            image = Image.fromarray((np.array(Image.open(target_dir + filename))>>8).astype(np.uint8)).convert('RGB')
            image.save(jpg_file_path + filename[:-3] + "jpg")
        else:
            print("error" + filename)
    pass
def main():
    path = 'C:\\Users\\jenni\\Desktop\\Diss_Work\\Unbroken\\'
    # the count should be the number of files in the other class to match
    # In my case, I matched the 8689 of single fracture breaks
    #count_to_match = 14158
    #augment_data(path, count_to_match)

    # Now before passing images to the neural networks, they have to 
    # be converted to JPGs for pytorch to handle (since it uses the PIL library)
    # The dataset I'm using has original images of 16-bit greyscale PNG images, which
    # require more adjustments to correctly handle the greyscale nature of them.
    # However, augmented images pre-handle this, and can be converted to 
    # JPG easier than the original images. 
    # convert_images_to_jpg switches on if there's an added chunk to the filename
    # as there would be for augmented images.
    target_fractured = 'C:\\Users\\jenni\\Desktop\\Diss_Work\\Fractured_w_multi\\'
    target_unbroken = 'C:\\Users\\jenni\\Desktop\\Diss_Work\\Unbroken_w_multi\\'
    jpg_dir_fractured = 'C:\\Users\\jenni\\Desktop\\Diss_Work\\jpgwmulti\\Fractured\\'
    jpg_dir_unbroken = 'C:\\Users\\jenni\\Desktop\\Diss_Work\\jpgwmulti\\NotFractured\\'
    #convert_images_to_jpg(target_dir=target_fractured, jpg_dir=jpg_dir_fractured)
    convert_images_to_jpg(target_dir=target_unbroken, jpg_dir=jpg_dir_unbroken)

if __name__ == "__main__":
    main()
