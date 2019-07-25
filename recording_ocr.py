# from mrcnn.config import Config
# from mrcnn import utils
# import mrcnn.model as modellib
# from mrcnn import visualize
# from mrcnn.model import log
from TextDetect import text_detect
import numpy as np
import cv2
from difflib import SequenceMatcher
from tesseract_wrapper import image_to_string
import requests 

arr = []
THREASHOLD = 0.3

# mask should be inthe form of [[x,y],[x,y],[x,y]]
# cover the background of the input
# future filter modification can be added here
def filter_image(img, mask):

    # Define the polygon coordinates to use or the crop
    # polygon = [[[20,110],[450,108],[340,420],[125,420]]]
    polygon = [mask]
    print(polygon)

    # First find the minX minY maxX and maxY of the polygon
    minX = img.shape[1]
    maxX = -1
    minY = img.shape[0]
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
    cropedImage = np.zeros_like(img)
    for y in range(0,img.shape[0]):
        for x in range(0, img.shape[1]):

            if x < minX or x > maxX or y < minY or y > maxY:
                continue

            if cv2.pointPolygonTest(np.asarray(polygon).astype(int),(x,y),False) >= 0:
                cropedImage[y, x, 0] = img[y, x, 0]
                cropedImage[y, x, 1] = img[y, x, 1]
                cropedImage[y, x, 2] = img[y, x, 2]

    # Now we can crop again just the envloping rectangle
    finalImage = cropedImage[int(minY):int(maxY),int(minX):int(maxX)]

    ave_x = 0
    ave_y = 0
    for (x, y) in mask:
        ave_x += x
        ave_y += y
    ave_x = ave_x / len(mask)
    ave_y = ave_y / len(mask)

    return finalImage, (ave_x, ave_y)



def merge(t1, t2, mismatch, occurance):
    insert = False
    score = 0
    matched = ''
    if not t1:
        ms = t2
        s = len(t2)
        o = [1] * s
        occurance = o
    elif not t2:
        ms = t1
        s = len(t1)
        o = [1] * s
        occurance = o
    elif t1[0] == t2[0]:
        # match this character
        matched = t1[0]
        if mismatch:
            occurance[0] += 1
        else:
            occurance[0] += 2
        ms, s, o = merge(t1[1:len(t1)], t2[1:len(t2)], False, occurance[1:len(occurance)])
        occurance = [occurance[0]] + o
    else:
        # this needs to be conformed
        if arr[len(t1)][len(t2)] is not None:
            ms, s, o = arr[len(t1)][len(t2)]
            occurance = o
        else:
            ms1, s1, o1 = merge(t1, t1[0] + t2, True, occurance)
            ms2, s2, o2 = merge(t2[0] + t1, t2, True, [0] + occurance)
            if s1 <= s2:
                s = s1 + 1
                ms = ms1
                o = o1
            else:
                s = s2 + 1
                ms = ms2
                o = o2
                insert = True
            occurance = o

    if not mismatch and arr[len(t1)][len(t2)] is None:
        arr[len(t1)][len(t2)] = (ms, s, occurance)
    matched += ms
    score += s
    return matched, score, occurance


def merge_text(text_array):
    # merge logic is: 
    # when merging, only add new characters
    # then, delete occurances that are lower than
    # certain number
    result = text_array[0]
    del text_array[0]
    occurance = [1] * len(result)
    for text in text_array:
        global arr
        arr = [[None] * (len(text) + 1)] * (len(result) + 1)
        result, _, occurance = merge(result, text, False, occurance)

    merged = ""
    # threashold for determining whether the character is a misread or
    # actual reading
    text_threashold = len(text_array)*1.0/3
    for c, n in zip(result, occurance):
        if n > text_threashold:
            merged += c

    return merged

# ((text, center_coor), text_id), frame_id
# (['read','reaed', 'rfeaed']) 
# ((('read', 1), 0), 0)
# ((('reaed', 1), 0), 1)
# ((('rfeaed', 1), 0), 2)
def merge_texts_array(texts, num):
    merged = []
    for x in range(0,num):
        merged.append({"all_text" : [], 'frame_id': []})
    text_arr = []
    # text_arr = [(frame_id, text_id, text) for (frame, frame_id) in enumerate(texts) for ((text, center_coor), text_id) in frame]
    for (frame_id, frame) in enumerate(texts):
        for ((text, center_coor), text_id) in frame:
            text_arr.append((frame_id, text_id, text))
    for frame_id, text_id, text in text_arr:
        merged[text_id]['all_text'].append(text.lower())
        merged[text_id]['frame_id'].append(frame_id)
    result = []
    for text in merged:
        # text['text'] = merge_text(text['all_text'])
        # text['frames'] = (min(text['frame_id']), max(text['frame_id']))
        # print(merge_text(text['all_text']))
        result.append({'text': merge_text(text['all_text']), 'start_frame': min(text['frame_id']), 'end_frame': max(text['frame_id'])})
        # del text['all_text']
        # del text['frame_id']
    return result

def similarity(t1, t2):
    t1, (x1, y1) = t1
    t2, (x2, y2) = t2
    dis = ((x1 - x2)**2 + (y1 - y2)**2) ** (1/2)
    text_dis = SequenceMatcher(None, t1, t2).ratio()
    return  - dis/100 + text_dis

# file should be a path to a 1s or few seconds long video
def video_to_text(file, model = None, lang = 'ENG'):
    vc = cv2.VideoCapture(file)
    # c is the frame counter
    c = 0
    texts = []
    if vc.isOpened():
        success, frame = vc.read()
    else:
        print('failed')
        # TODO exception

    # used this for accessing merged result
    text_id_incre = 0


    # constraint on c so that the program does not run for too long
    # iter all the frames
    while success and c < 500:
        print("running on frame ", c)
        # read one frame
        texts.append([])
        rval, frame = vc.read()

        # grab results from the detection model
        # results = model.detect([frame], verbose=1)
        # r = results[0]

        # grab the results from EAST server
        import requests 
        import os
         
        dirpath = os.getcwd()
          
        # Save image in set directory 
        url = "http://0.0.0.0:8769/"
        path = dirpath + '/outfile.jpg'
        rpath = dirpath + '/result'
        cv2.imwrite(path, frame)

        # data to be sent to server 
        files = [
            ('imagePath', path),
            ('resultPath', rpath)
        ]
        # sending post request and saving response as response object 
        r = requests.post(url = url, files = files) 

        # read detection result
        import pickle
        with open(rpath, 'rb') as f:
            detected = pickle.load(f)
            r = []
            for ele in detected['text_lines']:
                r.append(([ele['x0'], ele['y0']],[ele['x1'], ele['y1']],[ele['x2'], ele['y2']],[ele['x3'], ele['y3']]))

        # iter through each mask for this frame and apply
        # ocr
        identified = -1
        # for mask in r['mask']:
        for mask in r:
            filtered_img, center = filter_image(frame, mask)
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

                if identified is -1:
                    identified = text_id_incre
                    text_id_incre += 1

                texts[c].append(((text, center), identified))
                identified = -1

        c = c + 1
        cv2.waitKey(1)
    vc.release()
    # call merge text
    text = merge_texts_array(texts, text_id_incre)
    # TODO: ASR text and merge

    return text