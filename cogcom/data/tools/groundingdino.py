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

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span



def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_path, onbox=None):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
    ori_size = image_pil.size

    if onbox is not None:
        image_pil = image_pil.crop(onbox)

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image, ori_size


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(caption),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases


    return boxes_filt, pred_phrases

import nltk
def find_noun_phrases(caption: str) -> List[str]:
    if not caption:
        return [caption]

    def remove_punctuation(text):
        punct = ['|', ':', ';', '@', '(', ')', '[', ']', '{', '}', '^',
             '\'', '\"', '’', '`', '?', '$', '%', '#', '!', '&', '*', '+', ',', '.'
             ]
        for p in punct:
            text = text.replace(p, '')
        return text.strip()
    
    caption = caption.lower()
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)

    grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(pos_tags)

    noun_phrases = list()
    for subtree in result.subtrees():
        if subtree.label() == 'NP':
            noun_phrases.append(' '.join(t[0] for t in subtree.leaves()))

    noun_phrases = [remove_punctuation(phrase) for phrase in noun_phrases]
    noun_phrases = [phrase for phrase in noun_phrases if phrase != '']

    if len(noun_phrases) ==0: # avoid exception
        noun_phrases = [caption]

    return noun_phrases

def restore_boxes(boxes, img_size, onbox=None):
    """restore ration-based coordinates to the pixel-based ones"""
    size = img_size
    if onbox is not None:
        size = (onbox[2]-onbox[0], onbox[3]-onbox[1])
    rt_boxes = []
    for box in boxes:
        box[0] -= box[2]/2
        box[1] -= box[3]/2
        box[0] *= size[0]
        box[1] *= size[1]
        box[2] = box[0] + box[2]*size[0]
        box[3] = box[1] + box[3]*size[1]
        box = [round(b) for b in box]
        if onbox is not None:
            box = [b+ob for b,ob in zip(box, [onbox[0], onbox[1], onbox[0], onbox[1]])]
        rt_boxes.append(box)
    return rt_boxes


def get_positive_spans(caption: str, phrases: List[str]):
    positive_spans = []
    for phr in phrases:
        ph_s = ph_e = caption.find(phr)
        ph_spans = []
        for i, ch in enumerate(phr):
            if ch == ' ' or i==len(phr)-1:
                ph_e = ph_e+1 if i==len(phr)-1 else ph_e
                if ph_e!=ph_s:
                    ph_spans.append([ph_s, ph_e])
                ph_s = ph_e = ph_e+1
            else:
                ph_e += 1
        positive_spans.append(ph_spans)
    return positive_spans






class GroundingDINO:
    def __init__(self,config_file, checkpoint_path, box_threshold=0.3, text_threshold=0.25,cpu_only=False):
        self.cpu_only = cpu_only
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.model = load_model(config_file, checkpoint_path, cpu_only)

    def annoatate_grounding(self, image_path, onbox, caption, phrases):
        # load image
        image_pil, image, ori_size = load_image(image_path, onbox)
        token_spans = get_positive_spans(caption, phrases)

        # run model
        boxes_filt, pred_phrases = get_grounding_output(
            self.model, image, caption, self.box_threshold, self.text_threshold, cpu_only=self.cpu_only, token_spans=token_spans
        )
            
        boxes = restore_boxes(boxes_filt.tolist(), ori_size, onbox)
        # ex['img_size'] = ori_size
        # qa['steps_returns'][ret] = boxes
        return boxes

