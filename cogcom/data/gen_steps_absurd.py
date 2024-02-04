""" Given a language question for an image, find the core object that the question is askting for.
  e.g.,
    Q: What is the number of the train parking on the railway?
    CoreObj: (the train, the train parking on the railway)
"""
import os, sys
import re
import json
import urllib3
# import jsonlines
import random
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import multiprocessing
import itertools
import glob
from functools import partial
import time

from utils.template_util import *
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from tools.gpt4v import GPT4VInterface


TEMPLATE = 'Given a image and an absurd question about the given image (the question usually asks about non-existent objects in the picture), please generate a multi-step reasoning chain to refute the question. Please output the generation result as a json with the format of {"steps": [xxx, xxx, ...], "conclusion": xxx}.\nQuestion: QUESTION'

func = GPT4VInterface()



def process_one_line(data,):

    for qa in data['metadata']:
        if 'TDIUC' in qa['unique_id'] and qa.get('question_type', None) and qa.get('question_type').lower() =='absurd':
            question, answer, image_path = qa['question'], qa['answer'], data['image_path']

            # x-shot
            prompt = TEMPLATE.replace('QUESTION', question)

            
            status, result = func.get_response(prompt=prompt, image_path=image_path)
            max_calls = 10
            while status != 200 and max_calls > 0:
                status, result = func.get_response(prompt=prompt, image_path=image_path)
                max_calls -= 1
            if status != 200: # still failed
                print("Failed to call API.")
                return data

            # parser result
            try:
                result = result.replace('\n','')
                formatted_ptr = re.compile(r'.*?```json(.*?)```.*')
                if formatted_ptr.match(result):
                    result = formatted_ptr.match(result).group(1)
                ret_json = json.loads(result)
                qa['steps'] = ret_json['steps']
                qa['conclusion'] = ret_json['conclusion']
            except:
                print(f"Parsing result failed.")
    return data


def process_multi_lines(lines, save_f, rank=-1):
    result = []
    if rank == 0:
        lines = tqdm(lines, desc=time.asctime())
    with open(save_f, 'a') as fout:
        for data in tqdm(lines):
            new_data = process_one_line(data)
            result.append(new_data)
            fout.write(json.dumps(new_data) + '\n')
            fout.flush()
        # print(json.dumps(new_data)+'\n', file=fout, flush=True)
    return result



if __name__ == "__main__":

    data_dir = f"save/processed/TDIUC"
    save_dir = f"save/steps_absurd"
    os.makedirs(save_dir, exist_ok=True)
    # Resume
    finished_lines = {}
    for fname in  list(glob.glob(f'{save_dir}/*',recursive=True)):
        with open(fname) as f:
            finished_lines.update({json.loads(line)['image_path']: json.loads(line) for line in f.readlines()})
    print(f"{len(finished_lines)} items are already finished previously, which will be skipped. ")

    # Process all datasets
    train_results = []
    # train_files = list(glob.glob('training_data/*/*',recursive=True))
    train_files = list(glob.glob(f'{data_dir}/*',recursive=True))
    include = ['TDIUC']
    train_lines = []
    skipped = 0
    for file_name in train_files:
        if any([ds in file_name for ds in include]):
            assert '.json' in file_name
            if  not 'train.jsonl' in file_name:
                continue
            with open(file_name,'r') as fin:
                for line in fin:
                    line = json.loads(line)
                    if line['image_path'] in finished_lines: # skip the already finished lines
                        skipped += 1
                        continue
                    train_lines.append(line)
    assert skipped == len(finished_lines)
    # random.shuffle(train_lines) # shuffle for efficient
    num_process = min(4, len(train_lines))
    chunk_size = len(train_lines) // num_process + int(bool(len(train_lines) % num_process))
    chunk_src = [train_lines[i: i+chunk_size] for i in range(0, len(train_lines), chunk_size)]
    pool = multiprocessing.Pool(processes=num_process)
    for i in range(len(chunk_src)):
        # train_results.append(
        #         pool.apply_async(process_multi_lines, args=(chunk_src[i], ), kwds={"save_f": f"{save_dir}/{i}.jsonl", "rank": i})
        #     )
        process_multi_lines(chunk_src[i], f"{save_dir}/{i}.jsonl")
    pool.close(); pool.join()
    tot_train = sum([len(rt) for rt in train_results[i].get() for i in range(len(chunk_src))])
    print('Total training examples: %d' % tot_train)

