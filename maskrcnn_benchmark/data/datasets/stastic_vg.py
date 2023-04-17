import json
from tqdm import tqdm
import nltk
def process_region_descriptions(regs):
    boxes = []
    phrases = []
    pp = 0
    p =0
    for reg in regs:
        box_w = float(reg['width'])
        box_h = float(reg['height'])
        box_x = float(reg['x'])
        box_y = float(reg['y'])

        bot_right_x = box_x + box_w
        bot_right_y = box_y + box_h

        phrase = reg['phrase'].lower().strip()
        tokenized = nltk.word_tokenize(phrase)
        tagged = nltk.pos_tag(tokenized)
        jlk = 0
        kl = 0
        klp=0
        for i in tagged:
            if 'NN' in i[-1]:
                jlk += 1
                klp +=1
            if 'VB' in i[-1]:
                kl +=1
        if jlk ==1 and kl==1:
            pp += 1
        if klp >1:
            p +=1
        boxes.append([box_x, box_y, bot_right_x, bot_right_y])
        phrases.append(phrase)

    return p,pp, len(phrases)


def load_region_descriptions(path_file):
    print('loading region descrptions ....')
    reg_des = json.load(open(path_file, 'rb'))

    imgs = {}
    shot2=shot1 = 0
    p = 0
    for meta_im in tqdm(reg_des):
        s,ss,a = process_region_descriptions(meta_im['regions'])
        p += a
        shot2 += s
        shot1 += ss
        print(shot1*1./p, shot2*1.0/p)

load_region_descriptions('/mnt/hdd2/hetao/vg2/region_descriptions.json')