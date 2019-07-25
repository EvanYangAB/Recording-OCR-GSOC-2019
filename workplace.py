import requests 
import cv2 
import os
 
dirpath = os.getcwd()
  
# Save image in set directory 
# Read RGB image 
# img = cv2.imread('outfile.jpg') 
# print(dirpath + '/outfile.jpg')
# defining the api-endpoint  
url = "http://0.0.0.0:8769/"
path = dirpath + '/outfile.jpg'
rpath = dirpath + '/result'

# data to be sent to api 
files = [
    ('imagePath', path),
    ('resultPath', rpath)
]
  
# # sending post request and saving response as response object 
# r = requests.post(url = url, files = files) 

import pickle
with open(rpath, 'rb') as f:
    detected = pickle.load(f)
    result = []
    for ele in detected['text_lines']:
        # result.append([[ele['x0'] -20, ele['y0']-10],[ele['x1'] +530, ele['y1']- 20],[ele['x2'] + 550, ele['y2']],[ele['x3']-20, ele['y3']]])
        result.append([[ele['x0'] -20, ele['y0']],[ele['x1'] +530, ele['y1']],[ele['x2'] + 550, ele['y2']],[ele['x3']-20, ele['y3']]])
print(result[0])

import cv2
import numpy as np

# Read a image
I = cv2.imread('outfile.jpg')

# Define the polygon coordinates to use or the crop
# polygon = [[[20,110],[450,108],[340,420],[125,420]]]
polygon = [result[1]]
print(polygon)

# First find the minX minY maxX and maxY of the polygon
minX = I.shape[1]
maxX = -1
minY = I.shape[0]
maxY = -1
for point in polygon[0]:

    x = point[0]
    y = point[1]

    if x < minX:
        minX = x
    if x > maxX:
        maxX = x
    if y < minY:
        minY = y
    if y > maxY:
        maxY = y

# Go over the points in the image if thay are out side of the emclosing rectangle put zero
# if not check if thay are inside the polygon or not
cropedImage = np.zeros_like(I)
for y in range(0,I.shape[0]):
    for x in range(0, I.shape[1]):

        if x < minX or x > maxX or y < minY or y > maxY:
            continue

        if cv2.pointPolygonTest(np.asarray(polygon).astype(int),(x,y),False) >= 0:
            cropedImage[y, x, 0] = I[y, x, 0]
            cropedImage[y, x, 1] = I[y, x, 1]
            cropedImage[y, x, 2] = I[y, x, 2]

# Now we can crop again just the envloping rectangle
finalImage = cropedImage[int(minY):int(maxY),int(minX):int(maxX)]

cv2.imwrite('finalImage1.png',finalImage)
