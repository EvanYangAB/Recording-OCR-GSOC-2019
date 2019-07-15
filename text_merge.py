arr = []
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
        print(t1, t2)
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
    text_threashold = len(result)*1/3
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
    merged = [{"all_text" : [], 'frame_ids': []}] * num
    text_arr = []
    # text_arr = [(frame_id, text_id, text) for (frame, frame_id) in enumerate(texts) for ((text, center_coor), text_id) in frame]
    for (frame, frame_id) in enumerate(texts):
        print(frame, frame_id)
        for ((text, center_coor), text_id) in frame:
            text_arr.append((frame_id, text_id, text))
    for frame_id, text_id, text in text_arr:
        merged[text_id]['all_text'].append(text)
        merged[text_id]['frame_id'].append(frame_id)
    for text in merged:
        text['text'] = merge_text(text['all_text'])
        text['frames'] = (min(text['frame_id']), max(text['frame_id']))
        del text['all_text']
        del text['frame_id']
    return text