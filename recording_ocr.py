from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import numpy as np
import cv2
from tesseract_wrapper import image_to_string

# mask should be inthe form of [[x,y],[x,y],[x,y]]
# cover the background of the input
# future filter modification can be added here
def filter_image(img, mask):
    fter = np.zeros((img.shape[0], img.shape[1]))
    cv2.fillConvexPoly(fter, mask, 1)
    fter = fter.astype(np.bool)
    out = np.zeros_like(img)
    out[fter] = img[fter]
    return out

def merge_texts(texts):
    # TODO

# file should be a path to a 1s or few seconds long video
def video_to_text(file, model， lang):
    vc = cv2.VideoCapture(file)
    texts = []
    if vc.isOpened():
        rval , frame = vc.read()
    else:
        # TODO exception

    # iter all the frames
    while rval:
        # read one frame
        rval, frame = vc.read()
        # grab results from the detection model
        results = model.detect([frame], verbose=1)
        r = results[0]
        # iter through each mask for this frame and apply
        # ocr
        for mask in r['mask']:
            filtered_img = filter_image(original_image, mask)
            text.append(image_to_string(filtered_img, lang))

        c = c + 1
        cv2.waitKey(1)
    vc.release()
    # call merge text
    text = merge_texts(texts)
    # TODO: ASR text and merge

    return text