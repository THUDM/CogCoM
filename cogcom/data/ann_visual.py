import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import tqdm
from typing import List, Dict, Tuple
import re, json
import random
import collections

from tools.groundingdino import GroundingDINO, find_noun_phrases


def load_image(image_path, onbox=None):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
    ori_size = image_pil.size

    if onbox is not None:
        image_pil = image_pil.crop(onbox)

    # transform = T.Compose(
    #     [
    #         T.RandomResize([800], max_size=1333),
    #         T.ToTensor(),
    #         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #     ]
    # )
    # image, _ = transform(image_pil, None)  # 3, h, w
    # return image_pil, image, ori_size
    return image_pil, ori_size



from paddleocr import PaddleOCR, draw_ocr
ocr_tool = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
def get_ocr(img, onbox=None, ori_size=None):
     if onbox is not None: # using specified region !
        img = img.crop(onbox)
     img = np.array(img)
     result = ocr_tool.ocr(img, cls=True)
     new_result = []
     for rt in result:
        box = [*rt[0][0], *rt[0][2]]
        if onbox is not None:
            box = [b+off for b,off in zip(box, [ori_size[0],ori_size[1],ori_size[0],ori_size[1]])]
        new_result.append([box,  rt[1][0], rt[1][1]]) # [box, text, score]
     return new_result



from num2words import num2words
def is_equivalent_numbers(num1, num2):
    num1_vars = [num1]
    if isinstance(num1, int) or isinstance(num1, float):
        num1_vars.extend([num2words(num1, to='cardinal'), num2words(num1, to='ordinal'), num2words(num1, to='ordinal_num')])
    num2_vars = [num2]
    if isinstance(num2, int) or isinstance(num2, float):
        num2_vars.extend([num2words(num2, to='cardinal'), num2words(num2, to='ordinal'), num2words(num2, to='ordinal_num')])
    if set(num1_vars).intersection(set(num2_vars)).__len__() > 0:
        return True
    else:
        return False




from Levenshtein import distance as edit_distance
def annotate_ocr(image_path, onbox, answer=None):
    # load image
    image_pil, ori_size = load_image(image_path, onbox)
    ocr_res = get_ocr(image_pil, onbox, ori_size)

    found = False
    # simple rule to match ocr result with answer
    ocr_res_str = ""
    for iii in range(len(ocr_res)):
        res = ocr_res[iii][1]
        ocr_res_str = ocr_res_str + " " + ocr_res[iii][1]
        ocr_res_str = ocr_res_str.strip()
        if answer:
            dist = edit_distance(ocr_res_str, answer)
            if max(len(ocr_res_str), len(answer)) > 0 and float(dist) / max(len(answer), len(ocr_res_str))>=0.5:
                found = True
                break
    if found:
        ocr_res_str = answer
    return ocr_res, ocr_res_str, found



def annoatate_grounding(image_path, onbox,  caption, phrases):
    ret = groundingdino.annoatate_grounding(image_path, onbox, caption, phrases)
    return ret



PREVS = ["Using {} to ", "Based on {} to ", "Leveraging {} to ", "Utilizing {} to "]
CONJS = ['which is', 'resulting', 'and the result is']
def synthesize_com(func=None, phrase=None, param=None, variable=None, onbox=None, ret=None, ret_value=None, desc=None, found=False, 
                   add_mnp_first=True, replace_post=True):
    # synthesize com
    assert desc is not None
    if func is not None:
        try:
            # sep = re.findall(r',\s+.*?return', desc)
            # sep = re.findall(r'and return', desc)
            variables = {ret: ret_value}
            if variable:
                variables[variable] = onbox
            desc = desc.strip()
            if add_mnp_first:
                new_func = re.sub(r'_\d+', "", func)
                new_func = new_func.upper()
                # desc = desc.strip()
                desc = desc[0].lower() + desc[1:]
                desc = random.choice(PREVS).format(new_func + f'({param})') + desc
            if replace_post:
                sep = re.findall(r',\s+.*?return', desc)
                if len(sep)>0:
                    desc, _ = desc.split(sep[0])
                    desc = desc + ", {conj} `{ret}`.".format(conj=random.choice(CONJS), ret=ret)
            # if len(sep)>0:
            #     desc, post = desc.split(sep[0])
            #     # new_func = func.split('_')[0]
            #     new_func = re.sub(r'_\d+', "", func)
            #     new_func = new_func.upper()
            #     desc = desc.strip()
            #     desc = desc[0].lower() + desc[1:]
            #     desc = random.choice(PREVS).format(new_func + f'({param})') + desc
            #     desc += ", {conj} `{ret}`.".format(conj=random.choice(CONJS), ret=ret)
        except:
            desc = desc

        com = {
            'func': func,
            'param': phrase,
            'onbox': onbox,
            'variables': variables,
            'return': ret_value,
            'desc': desc,
            'found': found
        }
    else:
        com = {
            'func': func,
            'param': param,
            'onbox': onbox,
            'variables': None,
            'return': ret_value,
            'desc': desc,
            'found': found
        }
    return com



if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--in_file", "-i", type=str, default=None, required=True, help="input file"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default='com_outputs', required=True, help="output dir"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    input_file = args.in_file
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    out_f = os.path.join(output_dir, os.path.basename(input_file))

    groundingdino = GroundingDINO(args.config_file, args.checkpoint_path)

    with open(input_file) as f:
        dataset = list(map(json.loads, f.readlines()))
    results = []
    # tot_ground, tot_ocr= 0, 0
    tot_funcs = collections.defaultdict(int)
    # out_stream = open(out_f, 'w')
    for ex in tqdm.tqdm(dataset):
        image_path = ex['image_path']
        ex['img_size'] = Image.open(image_path).size
        for qa in ex['metadata']:
            qa['steps_returns'] = qa.get('steps_returns', {})
            qa['com_founds'] = qa.get('com_founds', [])
            # qa['final_com'] = qa.get('final_com', [])
            # qa['final_com'] = qa.get('final_com', {})
            final_com = {} # pointer
            if qa.get('steps', None):
                for i, step in enumerate(qa['steps']):
                    found = False
                    # ptr_func = re.compile(r'.*?\((.*?)\((.*?)\)->(.*?),(.*)')
                    ptr_func = re.compile(r'.*?\((.*?)\((.*?)\)->(.*?),(.*?)[\);]{0,2}$')
                    ptr_nofunc = re.compile(r'[(\s]{0,2}None[\s,]+(.*?)[\);]{0,2}$')

                    matched_func, matched_nofunc = False, False
                    if ptr_func.match(step) and len(ptr_func.match(step).groups())==4:
                        func, param, ret, desc = ptr_func.match(step).groups()
                        matched_func = True
                    elif ptr_nofunc.match(step):
                        func, param, ret, desc = None, None, None, ptr_nofunc.match(step).group(1)
                        matched_nofunc = True

                    fid = f'{i-1},*'
                    if matched_func:
                        # try:
                        # pos = param.find('`')\
                        pos = -1
                        var_ptr = re.compile(r'.*?(`\S+`).*')
                        if var_ptr.match(param):
                            pos = param.find(var_ptr.match(param).group(1))
                        onboxes = [None]
                        phrase = param
                        variable = None
                        if pos >=0:
                            variable = var_ptr.match(param).group(1)[1:-1]
                            phrase = param[:pos]
                            if 'bbx' in variable:
                                onboxes = qa['steps_returns'].get(variable, [None])
                        phrase = find_noun_phrases(phrase)[0] # use noun phrase
                        # except:
                        #         print("Parsing phrase failed with %s" % qa['unique_id'])
                        #         continue

                        if 'grounding' in func:
                            for ii, onbox in enumerate(onboxes):
                                try:
                                    boxes = annoatate_grounding(image_path, onbox, caption=phrase, phrases=[phrase])
                                except Exception as e:
                                    print(e)
                                    boxes = []
                                qa['steps_returns'][ret] = boxes
                                # get father: the `onbox` should in father's returns
                                for k,v in final_com.items():
                                    if onbox and isinstance(v['return'], list) and onbox in v['return']:
                                        _, fid = k.split('--')
                                        break
                                curid = f'{fid}--{i},{ii}'
                                # final_com[f'{i}-{ii}'] = synthesize_com(func, phrase, param, variable, onbox, ret, boxes, desc)
                                final_com[curid] = synthesize_com(func, phrase, param, variable, onbox, ret, boxes, desc)
                        elif 'OCR' in func:
                            for ii, onbox in enumerate(onboxes):
                                ocr_res, ocr_res_str, found = annotate_ocr(image_path, onbox, qa['answer'])
                                qa['steps_returns'][ret] = ocr_res_str
                                # get father: the `onbox` should in father's returns
                                for k,v in final_com.items():
                                    if onbox and isinstance(v['return'], list) and onbox in v['return']:
                                        _, fid = k.split('--')
                                        break
                                curid = f'{fid}--{i},{ii}'
                                if found:
                                    # qa['com_founds'].append(f'{i}-{ii}')
                                    qa['com_founds'].append(curid)
                                # final_com[f'{i}-{ii}'] = synthesize_com(func, phrase, param, variable, onbox, ret, ocr_res_str, desc, found=found)
                                final_com[curid] = synthesize_com(func, phrase, param, variable, onbox, ret, ocr_res_str, desc, found=found)
                        elif 'counting' in func:
                            ii = 0
                            # get father: the `onboxes` should equals to father's returns
                            for k,v in final_com.items():
                                if onboxes and onboxes[0] is not None and isinstance(v['return'], list) and onboxes == v['return']:
                                    _, fid = k.split('--')
                                    break
                            curid = f'{fid}--{i},{ii}'
                            if onboxes and onboxes[0] is not None:
                                ret_count = len(onboxes)
                                if is_equivalent_numbers(ret_count, qa['answer']):
                                    found = True
                                    # qa['com_founds'].append(f'{i}-{ii}')
                                    qa['com_founds'].append(curid)
                                    ret_count = qa['answer']
                            value = qa['steps_returns'].get(variable, None)
                            # final_com[f'{i}-{ii}'] = synthesize_com(func, phrase, param, variable, value, ret, qa['answer'], desc, found=found)
                            # final_com[curid] = synthesize_com(func, phrase, param, variable, value, ret, qa['answer'], desc, found=found)
                            final_com[curid] = synthesize_com(func, phrase, param, variable, value, ret, qa['answer'], desc, found=found, add_mnp_first=False)
                        else: # other manipulation， such as crop_and_zoom_in, calculate
                            for ii, onbox in enumerate(onboxes):
                                # get father: the `onbox` should in father's returns
                                for k,v in final_com.items():
                                    if onbox and isinstance(v['return'], list) and onbox in v['return']:
                                        _, fid = k.split('--')
                                        break
                                curid = f'{fid}--{i},{ii}'
                                # final_com[f'{i}-{ii}'] = synthesize_com(func, phrase, param, variable, onbox, ret, ret, desc, found=found)
                                final_com[curid] = synthesize_com(func, phrase, param, variable, onbox, ret, ret, desc, found=found)

                        pure_func = re.sub(r'_\d+', "", func)
                        tot_funcs[pure_func] += 1
                        tot_funcs['found_'+str(found)] += 1

                    elif matched_nofunc:
                        # find if have variables
                        # variables = None
                        # var_ptr = re.compile(r'.*?`(\S+)`.*')
                        # if var_ptr.match(desc):
                        #     variables = var_ptr.findall(desc)
                        #     variables = {va:qa['steps_returns'][va] for va in variables if qa['steps_returns'].get(va, None)}
                        # SOLUTION: the variables should be searched in ancestors of current chain

                        ii = 0
                        curid = f'{fid}--{i},{ii}'
                        if i == len(qa['steps'])-1: # found if at the last step
                            found = True
                            # qa['com_founds'].append(f'{i}-{ii}')
                            qa['com_founds'].append(curid)
                        # final_com[f'{i}-{ii}'] = synthesize_com(desc=desc, found=found)
                        final_com[curid] = synthesize_com(desc=desc, found=found)
            # qa['final_com'] = synthesize_com_tree(final_com)
            qa['final_com'] = final_com
        results.append(ex)
        # out_stream.write(json.dumps(ex)+'\n')
        # out_stream.flush()

    # # visualize pred
        # size = image_pil.size
        # pred_dict = {
        #     "boxes": boxes_filt,
        #     "size": [size[1], size[0]],  # H,W
        #     "labels": pred_phrases,
        # }
        # # import ipdb; ipdb.set_trace()
        # image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
        # image_with_box.save(os.path.join(output_dir, "pred.jpg"))

    # print(f"Total #ground={tot_ground}, #OCR={tot_ocr}")
    for k,v in tot_funcs.items():
        print(f'{k}: {v}')
    # out_stream.close()
    with open(out_f, 'w') as f:
        for line in results:
            f.write(json.dumps(line) + '\n')

