# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause
# Import datasets, classifiers and performance metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
from prepare_data import process_data, process_image, reduce_noise
import joblib
import numpy as np
import cv2

chars_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
chars_dict = {c: chars_list.index(c) for c in chars_list}

def train():
    # Create a classifier: a support vector classifier
    clf = svm.SVC(kernel="linear", C=10e5)
    images, labels = process_data("data/chars/")
    # Split data into 80% train and 20% test subsets
    images_train, images_test, labels_train, labels_test = train_test_split(
        images, labels, test_size=0.2, random_state=42)

    # Learn the digits on the train subset
    clf.fit(images_train, np.ravel(labels_train))
    print(clf.score(images_test, labels_test))
    joblib.dump(clf, "svm.pkl")
    print("Saved model to svm.pkl")


def predict_char(image_path):
    image = process_image(image_path).reshape(1, -1)
    print(image)
    clf = joblib.load("svm.pkl")
    actual = chars_list[clf.predict(image)[0]]
    return actual

def predict_string(file_path):
    res = ''
    dir = reduce_noise(file_path)

    img = cv2.imread()


    return res

if __name__=='__main__':
    print(predict_string('data/raw/2222.jpg'))

