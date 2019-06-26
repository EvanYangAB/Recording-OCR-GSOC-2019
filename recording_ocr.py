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

    # calculate centroid of masked area
    ave_x = 0
    ave_y = 0
    for (x, y) in mask:
        ave_x += x
        ave_y += y
    ave_x = ave_x / len(mask)
    ave_y = ave_y / len(mask)

    return out, (ave_x, ave_y)

# each element in the array in is in the form of
# represents a frame, each result in the frame is 
# in the form of: 
# (text, center coor), text id
def merge_texts(texts):


    # TODO

# file should be a path to a 1s or few seconds long video
def video_to_text(file, model，lang):
    vc = cv2.VideoCapture(file)
    texts = []
    if vc.isOpened():
        rval , frame = vc.read()
    else:
        # TODO exception

    # used this for accessing merged result
    text_id_incre = 0
    # iter all the frames
    while rval:
        # read one frame
        texts.append([])
        rval, frame = vc.read()
        # grab results from the detection model
        results = model.detect([frame], verbose=1)
        r = results[0]
        # iter through each mask for this frame and apply
        # ocr
        identified = -1
        for mask in r['mask']:
            filtered_img, center = filter_image(original_image, mask)
            text = (image_to_string(filtered_img, lang))
            # (text, center coordinate, frame #, indicator)
            # indicator is reserved for merging
            if c is not 0:
                # compare to the result in the previous frame, and merge
                # put texts that should be the same together

                # result is in the from of (text, center coor)
                # get the first one that has similarity score greater than
                # THREASHOLD
                for result, text_id in texts[c-1]:
                    if similarity(result, (text, center)) > THREASHOLD:
                        identified = text_id
                        break

                if identified = -1:
                    identified = text_id_incre
                    text_id_incre += 1

                texts[c].append(((text, center), identified))

        c = c + 1
        cv2.waitKey(1)
    vc.release()
    # call merge text
    text = merge_texts(texts)
    # TODO: ASR text and merge

    return text