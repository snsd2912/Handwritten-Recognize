import string
import numpy as np
import cv2
import sys
from PIL import Image
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
from sklearn.neighbors import KNeighborsClassifier
import joblib 
import pickle
from base.base import KNN_MODEL, pixels_to_hog_20

DIGIT_DIM = 20 # size of each digit is SZ x SZ
CLASS_N = 26 # a-z
QUANTITY = 10 
USER_IMG = "test_image.jpg"
WHITE = [255,255,255]
OUTFILE = "model.pkl"

def contains(r1, r2):
    r1_x1 = r1[0]
    r1_y1 = r1[1]
    r2_x1 = r2[0]
    r2_y1 = r2[1]
    
    r1_x2 = r1[0]+r1[2]
    r1_y2 = r1[1]+r1[3]
    r2_x2 = r2[0]+r2[2]
    r2_y2 = r2[1]+r2[3]
    
    #does r1 contain r2?
    return r1_x1 < r2_x1 < r2_x2 < r1_x2 and r1_y1 < r2_y1 < r2_y2 < r1_y2

def get_digits(contours):
    digit_rects = [cv2.boundingRect(ctr) for ctr in contours]   
    rects_final = digit_rects[:]

    for r in digit_rects:
        x,y,w,h = r
        if w < 10 and h < 10:        #too small, remove it
            rects_final.remove(r)    
    
    for r1 in digit_rects:
        for r2 in digit_rects:
            if contains(r1,r2) and (r2 in rects_final):
                rects_final.remove(r2)
    return rects_final


def proc_user_img(fn, model):
    print('loading "%s for digit recognition" ...' % fn)
    im = cv2.imread(fn)
    im_original = cv2.imread(fn)
    # pre-process the image by resizing it, converting it to
    # graycale, blurring it, and computing an edge map
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Output2", gray)
    # cv2.waitKey(0)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.GaussianBlur()
    # cv2.imshow("Output2", blurred)
    # cv2.waitKey(0)
    edged = cv2.Canny(blurred, 50, 200)
    cv2.imshow("Output2", edged)
    cv2.waitKey(0)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits_rect = get_digits(cnts)

    # for c in digitCnts:
    for rect in digits_rect:
        print("----------")
        x,y,w,h = rect
        # x,y,w,h = cv2.boundingRect(c)
        _ = cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)

        im_digit = im_original[y:y+h,x:x+w]
            
        thresh = 110
        im_digit = cv2.cvtColor(im_digit,cv2.COLOR_BGR2GRAY)
        im_digit = cv2.threshold(im_digit, thresh, 255, cv2.THRESH_BINARY)[1]
        # cv2.imshow("Output2", im_digit)
        # cv2.waitKey(0)
        im_digit = cv2.copyMakeBorder(im_digit,5,5,5,5,cv2.BORDER_CONSTANT,value=WHITE)
        
        im_digit = np.array(Image.fromarray(im_digit).resize(size=(DIGIT_DIM,DIGIT_DIM)))
        # cv2.imshow("Output2", im_digit)
        # cv2.waitKey(0)
        #       
        hog_img_data = pixels_to_hog_20([im_digit])  
        
        pred = model.predict(hog_img_data)
        print(pred)
        
        _ = cv2.putText(im, str(pred), (x,y),cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

        
    cv2.imwrite("result.png",im)  
    cv2.destroyAllWindows()

if __name__ == '__main__':

    with open(OUTFILE, 'rb') as file:  
        model = pickle.load(file)
        
    proc_user_img(USER_IMG,model)