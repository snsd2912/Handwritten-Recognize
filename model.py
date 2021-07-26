import string
import numpy as np
import cv2
import sys
import pickle
from base.base import KNN_MODEL, pixels_to_hog_20

DIGIT_DIM = 20 # size of each character is SZ x SZ
CLASS_N = 26 # a-z
QUANTITY = 10 
OUTFILE = "model.pkl"

def split2d():
    cells = np.empty((0,20,20))
    for i in range(CLASS_N):
        for j in range(QUANTITY):
            img_name = "./data/" + string.ascii_lowercase[i:i+1] + str(j) + ".png"
            img = cv2.imread(img_name, 0)
            # print(img.shape)
            cells = np.append(cells, [img[:,:]], axis = 0)
    return cells

def load_digits():
    digits = split2d()
    labels = np.empty(0)
    for i in range(CLASS_N):
        labels = np.append(labels, np.repeat(string.ascii_lowercase[i:i+1], 10))
    return digits, labels         


if __name__ == '__main__':
    digits, labels = load_digits()
    print('training ....')

    train_digits_data = pixels_to_hog_20(digits)
    train_digits_labels = labels
    
    # print('training KNearest...')  
    model = KNN_MODEL()
    model.train(train_digits_data, train_digits_labels)
  
    with open(OUTFILE, 'wb') as pickle_file:
        pickle.dump(model, pickle_file)
    
