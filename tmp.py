import cv2
from scipy import ndimage
import numpy as np
from skimage import data, morphology, transform, filter
import time

coins = data.coins()
coins_10_12 = (255*transform.resize(coins, (1000, 1200))).astype("uint8")
kernel = np.ones((3, 3)).astype("uint8")
kernel50 = np.ones((50, 50)).astype("uint8")
time1=time.clock()
cv2.dilate(coins, kernel, iterations = 1)
time2=time.clock()
print time2-time1
