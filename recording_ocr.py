from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import numpy as np
import cv2
from difflib import SequenceMatcher
from tesseract_wrapper import image_to_string

arr = []
THREASHOLD = 0.8

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


# occurance in the length of t1
def merge(t1, t2, mismatch, occurance):
    score = 0
    if t1[0] == t2[0]:
        # match this character
        matched = t1[0]
        if mismatch:
            occurance[0] += 1
        else:
            occurance[0] += 2
        ms, s, o = merge(t1[1:len(t1)], t2[1:len(t2)], False, occurance[1:len(occurance)])
    else:
        # this needs to be conformed
        if arr[len(t1)][len(t2)] is not None:
            ms, s, o = arr[len(t1)][len(t2)]
        else:
            ms1, s1, o1 = merge(t1, t1[0] + t2, True, occurance)
            ms2, s2, o2 = merge(t2[0] + t1, t2, True, [0] + occurance)
            if s1 > s2:
                s = s1
                ms = ms1
                o = o1
            else:
                s = s2
                ms = ms2
                o = o2
    if arr[len(t1)][len(t2)] is None:
        arr[len(t1)][len(t2)] = (ms, s, o)
    matched += ms
    score += s
    occurance = occurance[0] + o

    return matched, score, occurance


def merge_text(text_array):
    # merge logic is: 
    # when merging, only add new characters
    # then, delete occurances that are lower than
    # certain number
    result = text_array[0]
    del result[0]
    occurance = [1] * len(result)
    for text in text_array:
        arr = []
        result, _, occurance = merge(result, text, False, occurance)

    merged = ""
    # threashold for determining whether the character is a misread or
    # actual reading
    text_threashold = len(result)*1/3
    for c, n in zip(result, occurance):
        if n > text_threashold:
            merged += c

    return merged

# each element in the array in is in the form of
# represents a frame, each result in the frame is 
# in the form of: 
# (text, center coor), text id
def merge_texts_array(texts, num):
    merged = [{"all_text" : [], 'frame_ids': []}] * num
    text_arr = [(frame_id, text_id, text) for (frame, frame_id) in enumerate(texts) for (text, center_coor), text_id in frame]
    for frame_id, text_id, text in text_arr:
        merged[text_id]['all_text'].append(text)
        merged[text_id]['frame_id'].append(frame_id)
    for text in merged:
        text['text'] = merge_text(text['all_text'])
        text['frames'] = (min(text['frame_id']), max(text['frame_id']))
        del text['all_text']
        del text['frame_id']
    return text

def similarity(t1, t2):
    t1, (x1, y1) = t1
    t2, (x2, y2) = t2
    dis = ((x1 - x2)**2 + (y1 - y2)**2) ** (1/2)
    text_dis = SequenceMatcher(None, t1, t2).ratio()
    return dis/4 + text_dis

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
    text = merge_texts_array(texts, text_id_incre)
    # TODO: ASR text and merge

    return text