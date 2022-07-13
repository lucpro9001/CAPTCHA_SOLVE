from __future__ import division
from __future__ import print_function
import shutil
import os
from os import listdir
from os.path import join, isfile
from turtle import right
from PIL import Image, ImageChops
import numpy as np
import cv2
import random
import string
from imageio.v3 import imread

chars_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
chars_dict = {c: chars_list.index(c) for c in chars_list}

IMAGE_TOTAL = 5000
RAW_PATH = "data/raw/"
DENOISE_PATH = "data/denoise/"
SLICED_PATH = "data/sliced/"
AJUST_PATH = "data/ajust/"

part = 0
list_chars = [f for f in listdir('data/chars') if isfile(join('data/chars', f)) and 'jpg' in f]

def process_directory(directory):
    file_list = []
    for file_name in listdir(directory):
        file_path = join(directory, file_name)
        if isfile(file_path) and 'jpg' in file_name:
            file_list.append(file_path)
    return file_list

def process_image(image_path):
    image = imread(image_path)
    image = image.reshape(1080,)
    return np.array([x/255. for x in image])
def process_data(directory):
    images = []
    labels = []
    image_list = process_directory(directory)
    for image_path in image_list:
        images.append(process_image(image_path))
        labels.append(chars_dict[image_path.split('/')[-1].split('-')[0]])
    return np.array(images), np.array(labels).reshape([len(labels), 1])
def reduce_noise(file_path):
    img = cv2.imread(file_path)
    dst = cv2.fastNlMeansDenoisingColored(img,None,50,50,7,21)
    cv2.imwrite(file_path, dst)
    img = Image.open(file_path).convert('L')
    img = img.point(lambda x: 0 if x<128 else 255, '1')
    img.save(file_path)
    return file_path
def reduce_noise_dir(indir, outdir):
    # take file jpg in input directory
    list_file = process_directory(indir)
    for file_path in list_file:
        print(file_path)
        img = cv2.imread(file_path)
        dst = cv2.fastNlMeansDenoisingColored(img,None,50,50,7,21)
        new_location = join(outdir, file_path.split('/')[-1])
        cv2.imwrite(new_location, dst)
        img = Image.open(new_location).convert('L')
        img = img.point(lambda x: 0 if x<128 else 255, '1')
        img.save(new_location)
def crop(file_path, out_directory):
    part = 0
    img = Image.open(file_path)
    p = img.convert('P')
    w, h = p.size

    letters = []
    left, right= -1, -1
    found = False
    for i in range(w):
        in_letter = False
        for j in range(h):
            if p.getpixel((i,j)) == 0:
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
    for [l,r] in letters:
        if r-l < 40:
            bbox = (l, 0, r, h)
            crop = img.crop(bbox)
            crop = crop.resize((30,60))
            crop.save(join(out_directory, '{0:04}_{1}.jpg'.format(part, origin)))
            part += 1
def crop_dir(raw_directory, out_directory):
	list_file = process_directory(raw_directory)
	global part
	for file_path in list_file:
		print(file_path)
		img = Image.open(file_path)
		p = img.convert('P')
		w, h = p.size

		letters = []
		left, right= -1, -1
		found = False
		for i in range(w):
			in_letter = False
			for j in range(h):
				if p.getpixel((i,j)) == 0:
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
		for [l,r] in letters:
			if r-l < 40:
				bbox = (l, 0, r, h)
				crop = img.crop(bbox)
				crop = crop.resize((30,60))
				crop.save(join(out_directory, '{0:04}_{1}.jpg'.format(part, origin)))
				part += 1



def adjust_dir(directory, out_path):
    list_file = process_directory(directory)
    for file_path in list_file:
        crop_a_picture(file_path, out_path)

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
        i-=1
    # check first top row is black
    for i in range(len(data)):
        if is_row_lower_value(data[i], NEAREST_BLACK):
            data = data[i:]
            break
    # check first right column is black
    i = len(data[0]) - 1
    while i > 0:
        if(is_column_lower_value(data[:, i], NEAREST_BLACK)):
            data = data[:,:i]
            break
        i-=1
    # check first left column is black
    for i in range(len(data[0])):
        if(is_column_lower_value(data[:, i], NEAREST_BLACK)):
            data = data[:, i:]
            break
    image = Image.fromarray(data)
    image = image.resize((30, 36))		
    image.save(join(out_path, file_path.split('/')[-1]))	
    

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
def is_column_equal_value(column, val):
    for e in column:
        if e == val:
            return True
    return False

def determine_char(image_path):
    image = Image.open(image_path)
    data = np.asarray(image)
    

    print(data)

def bound_char(data, l):
    left, right = 0, 0
    n = len(data[0])
    for x in range(l, n):
        if(is_column_lower_value(data[:,x], 0)):
            left = x
            break
    for x in range(left+1, n):
        if is_column_lower_value(data[:,x], 0) and not is_column_equal_value(data[:,x], 0):
            right = x
            break
    return left, right

def rand_string(N=6):
	return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(N))

def rename(path, letter):
    new_name = letter+'-' + rand_string() + '.jpg'
    os.rename(path, join('data/chars/', new_name))
    return new_name
class Fit:
    letter = None
    difference = 0
def detect_char(path, filename):
    best = Fit()
    _img = Image.open(join(path, filename))
    for img_name in list_chars:
        current = Fit()
        img = Image.open(join('data/chars', img_name))
        current.letter = img_name.split('-')[0]
        difference = ImageChops.difference(_img, img)
        for x in range(difference.size[0]):
            for y in range(difference.size[1]):
                current.difference += difference.getpixel((x, y))/255.
        if not best.letter or best.difference > current.difference:
            best = current
    print(filename, best.letter)
    re_path = shutil.copy(join(path, filename), 'data/chars/')
    new_name = rename(re_path, best.letter)
    list_chars.append(new_name)

def detect_dir(directory):
	for f in listdir(directory):
		if isfile(join(directory, f)) and 'jpg' in f:
			detect_char(directory, f)

if __name__=='__main__':
    # reduce_noise_dir(RAW_PATH, DENOISE_PATH)
    # crop_dir(DENOISE_PATH, SLICED_PATH)
    # adjust_dir(SLICED_PATH, AJUST_PATH)
    # adjust_dir('data/ajust1/','data/chars/')
    # determine_char('data/raw/0001.jpg')

    # crop('data/raw/0001.jpg', 'tmp/null/')
    # adjust_dir('tmp/null/', 'tmp/null/')
    # crop_a_picture('tmp/null/0000_0002.jpg', 'tmp/null/')
    reduce_noise('0037.jpg')
    pass