
import cv2

from os import remove
from PIL import Image
import glob
import joblib
import numpy as np
from imageio.v3 import imread
from os.path import join, isfile
from os import listdir
import os

basedir = os.path.dirname(__file__)

chars_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
chars_dict = {c: chars_list.index(c) for c in chars_list}

winSize = (30, 36)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True


def process_image(image_path, cellSize):
    blockSize = (cellSize[0]*2, cellSize[1]*2)
    blockStride = cellSize
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride,
                            cellSize, nbins, derivAperture,
                            winSigma, histogramNormType, L2HysThreshold,
                            gammaCorrection, nlevels, signedGradients)

    descriptor = hog.compute(img)
    return np.array(descriptor)


def process_data(directory, cellSize):
    images = []
    labels = []
    image_list = process_directory(directory)
    for image_path in image_list:
        images.append(process_image(image_path, cellSize))
        labels.append(chars_dict[image_path.split('/')[-1].split('-')[0]])
    return np.array(images), np.array(labels)


def process_directory(directory):
    file_list = []
    for file_name in listdir(directory):
        file_path = join(directory, file_name)
        if isfile(file_path) and 'jpg' in file_name:
            file_list.append(file_path)
    return file_list


def reduce_noise(file_path):
    img = cv2.imread(file_path)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 50, 50, 7, 21)
    cv2.imwrite(file_path, dst)
    img = Image.open(file_path).convert('L')
    img = img.point(lambda x: 0 if x < 128 else 255, '1')
    img.save(file_path)
    return file_path


def crop(file_path, out_directory):
    part = 0
    img = Image.open(file_path)
    p = img.convert('P')
    w, h = p.size

    letters = []
    left, right = -1, -1
    found = False
    for i in range(w):
        in_letter = False
        for j in range(h):
            if p.getpixel((i, j)) == 0:
                in_letter = True
                break
        if not found and in_letter:
            found = True
            left = i
        if found and not in_letter and i-left > 25:
            found = False
            right = i
            letters.append([left, right])
    origin = file_path.split('/')[-1].split('.')[0]
    for [l, r] in letters:
        if r-l < 40:
            bbox = (l, 0, r, h)
            crop = img.crop(bbox)
            crop = crop.resize((30, 60))
            crop.save(
                join(out_directory, '{0:04}_{1}.jpg'.format(part, origin)))
            part += 1


def is_row_lower_value(row, val):
    for e in row:
        if e <= val:
            return True
    return False


def is_column_lower_value(column, val):
    for e in column:
        if e <= val:
            return True
    return False


def crop_a_picture(file_path, out_path):
    NEAREST_BLACK = 50
    # load the image
    image = Image.open(file_path)
    # convert image to numpy array
    data = np.asarray(image)
    # check first bottom row is black
    i = len(data) - 1
    while i > 0:
        if(is_row_lower_value(data[i], NEAREST_BLACK)):
            data = data[:i]
            break
        i -= 1
    # check first top row is black
    for i in range(len(data)):
        if is_row_lower_value(data[i], NEAREST_BLACK):
            data = data[i:]
            break
    # check first right column is black
    i = len(data[0]) - 1
    while i > 0:
        if(is_column_lower_value(data[:, i], NEAREST_BLACK)):
            data = data[:, :i]
            break
        i -= 1
    # check first left column is black
    for i in range(len(data[0])):
        if(is_column_lower_value(data[:, i], NEAREST_BLACK)):
            data = data[:, i:]
            break
    image = Image.fromarray(data)
    image = image.resize((30, 36))
    image.save(join(out_path, file_path.split('/')[-1]))


def adjust_dir(directory, out_path):
    list_file = process_directory(directory)
    for file_path in list_file:
        crop_a_picture(file_path, out_path)


def predict_char(image_path):
    image = process_image(image_path, (15, 18)).reshape(1, -1)
    clf = joblib.load("rfc.pkl")
    actual = chars_list[clf.predict(image)[0]]
    return actual


def predict_string(file_path):
    res = ''
    out_path = 'tmp/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    dir = reduce_noise(file_path)
    crop(dir, out_path)
    adjust_dir(out_path, out_path)
    file_list = process_directory(out_path)
    for f in sorted(file_list):
        res += predict_char(f)
    filelist = glob.glob(join(out_path, "*"))
    for f in filelist:
        remove(f)
    return res


if __name__=='__main__':
    print(predict_string('./data/raw/0033.jpg'))

