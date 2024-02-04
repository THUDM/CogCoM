""" Generate solving steps based on LLM.
@File    :   gen_steps.py
@Time    :   2024/2/4
@Author  :   Ji Qi 
@Contact :   qj20@mails.tsinghua.edu.cn
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

from tools.gpt4 import GPT4PI
from utils.template_util import *
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)





def process_one_line(data, func, shot):

    for qa in data['metadata']:
        if not qa.get('steps_txt', None):
            if 'TDIUC' in qa['unique_id'] and qa.get('question_type', None):
                if qa.get('question_type').lower() in ['absurd', 'object_presence']:
                    continue # skip absurd questions
                
            question, answer = qa['question'], qa['answer']

            # x-shot
            prompt = get_prompt(question, shot=shot)

            # server = 'chatglm2-66b'
            status, result = func(prompt=prompt)
            max_calls = 10
            while status != 200 and max_calls > 0:
                status, result = func(prompt=prompt)
                max_calls -= 1
            if status != 200: # still failed
                print("Failed to call API.")
                return data

            # parser result
            rt_steps = []
            try:
                out_steps = re.findall(r'Step\s+[\d+]:', result, re.IGNORECASE)
                for i, stp in enumerate(out_steps):
                    pos_s = result.find(stp)+ len(stp)
                    if i == len(out_steps)-1:
                        pos_e = len(result)
                    else:
                        pos_e = result.find(out_steps[i+1])
                    content = result[pos_s : pos_e]
                    rt_steps.append(content)
                qa['steps'] = rt_steps
            except:
                print(f"Parsing result failed.")

            qa['steps_txt'] = result
    return data


def process_multi_lines(lines, func, shot, save_f, rank=-1):
    result = []
    if rank == 0:
        lines = tqdm(lines, desc=time.asctime())
    with open(save_f, 'a') as fout:
        for data in tqdm(lines):
            new_data = process_one_line(data, func, shot)
            result.append(new_data)
            fout.write(json.dumps(new_data) + '\n')
            fout.flush()
        # print(json.dumps(new_data)+'\n', file=fout, flush=True)
    return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="save/processed")
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--split', type=str, default="train")
    args = parser.parse_args()

    func = GPT4PI.get_response

    data_dir = f"save/steps_{args.shot}shot"
    save_dir = f"save/steps_{args.shot}shot"
    os.makedirs(save_dir, exist_ok=True)
    # Resume
    finished_lines = {}
    for fname in  list(glob.glob(f'{save_dir}/*',recursive=True)):
        with open(fname) as f:
            finished_lines.update({json.loads(line)['image_path']: json.loads(line) for line in f.readlines()})
    print(f"{len(finished_lines)} items are already finished previously, which will be skipped. ")

    # Process all datasets
    train_results = []
    train_files = list(glob.glob(f'{args.data_dir}/*/*',recursive=True))
    include = ['ST-VQA', 'TextVQA', 'TDIUC']
    train_lines = []
    skipped = 0
    for file_name in train_files:
        if any([ds in file_name for ds in include]):
            assert '.json' in file_name
            # if  not 'train.jsonl' in file_name:
            if args.split not in os.path.basename(file_name):
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
    num_process = min(5, len(train_lines))
    chunk_size = len(train_lines) // num_process + int(bool(len(train_lines) % num_process))
    chunk_src = [train_lines[i: i+chunk_size] for i in range(0, len(train_lines), chunk_size)]
    pool = multiprocessing.Pool(processes=num_process)
    for i in range(len(chunk_src)):
        train_results.append(
                pool.apply_async(process_multi_lines, args=(chunk_src[i], ), kwds={"func":func, "shot":args.shot, "save_f": f"{save_dir}/{i}.jsonl", "rank": i})
            )
        # process_multi_lines(chunk_src[i], func, args.shot, f"{save_dir}/{i}.jsonl")
    pool.close(); pool.join()
    tot_train = sum([len(rt) for rt in train_results[i].get() for i in range(len(chunk_src))])
    print('Total training examples: %d' % tot_train)

