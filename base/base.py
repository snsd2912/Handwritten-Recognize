from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KNN_MODEL():                
    def __init__(self, k = 7):
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=k, p=2)



    def train(self, samples, responses):
        self.model.fit(samples, responses)

    def predict(self, samples):
        predicted = self.model.predict(samples)
        return predicted

def pixels_to_hog_20(pixel_array):
    hog_featuresData = []
    for img in pixel_array:
        #img = 20x20
        fd = hog(img, orientations=9, pixels_per_cell=(10,10), cells_per_block=(1,1), visualize=False)
        # print(fd)
        hog_featuresData.append(fd)
    hog_features = np.array(hog_featuresData, 'float64')
    print(hog_features)
    return np.float64(hog_features)