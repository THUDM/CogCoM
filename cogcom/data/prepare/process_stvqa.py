"""answer question base on the content of the image
@File    :   process_stvqa.py
@Time    :   2024/2/4
@Author  :   Ji Qi 
@Contact :   qj20@mails.tsinghua.edu.cn
"""
from argparse import ArgumentParser
import json
from pathlib import Path
import re
import random
import os
from os.path import exists
import tqdm
from collections import Counter


class InstructData:

    def __init__(self, ) -> None:
        self.data_dir = {
            'train': [f'save/raw/ST-VQA/train_task_{i}.json' for i in range(1,4)],
            'test': [f'save/raw/ST-VQA/test_task_{i}.json' for i in range(1,4)]}
        self.img_dir = 'save/raw/ST-VQA'
        self.out_data_dir = Path('save/processed/ST-VQA')
        self.splits = ['train', 'test']
        self.max_nums = [-1, -1]

    def create_data(self, ):
        for sid, split in enumerate(self.splits):
            for filename in self.data_dir[split]:
                data = []
                with open(filename, "r", encoding="utf-8") as fp:
                        data.extend(json.load(fp)['data'])
            question_ids = set()

            all_results = {}
            drop_num, tot_item_num = 0, 0
            for c_data in tqdm.tqdm(data):
                if c_data["set_name"] != split:
                    continue
                if 'answers' not in c_data:
                    answer = ''
                    answer_list = []
                else:
                    counts = Counter(c_data["answers"])
                    answer = sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]
                    answer_list = c_data["answers"]

                image_path = os.path.join(self.img_dir, c_data["file_path"])
                if not os.path.exists(image_path):
                    print(f"not found: {image_path}")
                    drop_num += 1
                    continue
                c_data = {
                    "unique_id": "ST-VQA-{}".format(c_data["question_id"]),
                    "question": c_data["question"],
                    "answer": answer,
                    "answer_list": answer_list,
                    # "image_path": image_path
                }
                if c_data["unique_id"] in question_ids:
                    print(f"[{split}]: find repeated question_ids, {c_data['unique_id']}")
                else:
                    question_ids.add(c_data["unique_id"])
                    all_results[image_path] = all_results.get(image_path, []) + [c_data]
                    tot_item_num += 1



            # save tarfiles
            save_path = os.path.join(self.out_data_dir, f"{split}.jsonl")
            os.makedirs(self.out_data_dir, exist_ok=True)
            item_num = 0
            with open(save_path, 'w') as f:
                max_count = self.max_nums[sid]
                for count, image_path in enumerate(all_results):
                    if count == max_count:
                        break
                    out_dict = {'image_path': image_path, 'metadata': all_results[image_path], }
                    f.write(json.dumps(out_dict) + '\n')
                    item_num += len(all_results[image_path])
            print(f"[ST-VQA - {split}]: #images/#total = {count}/{len(all_results)}, #QAs/#total={item_num}/{tot_item_num}, {drop_num} images are not found.")
    
                


if __name__ == '__main__':

    dataset = InstructData()
    dataset.create_data()