"""answer question base on the content of the image
@File    :   process_ocrvqa.py
@Time    :   2024/2/4
@Author  :   Ji Qi 
@Contact :   qj20@mails.tsinghua.edu.cn
"""
from argparse import ArgumentParser
import json
from pathlib import Path
import random
import os
from os.path import exists
import tqdm
from collections import Counter

class InstructData:

    def __init__(self, ) -> None:
        self.data_dir = {'train':'save/raw/TextVQA/train_val/TextVQA_0.5.1_train.json',
                    'val': 'save/raw/TextVQA/train_val/TextVQA_0.5.1_val.json',
                    'test': 'save/raw/TextVQA/test/TextVQA_0.5.1_test.json'}
        self.img_dir = {'train': 'save/raw/TextVQA/train_val/train_images',
                'val': 'save/raw/TextVQA/train_val/train_images',
                'test': 'save/raw/TextVQA/test/test_images'}
        self.splits = ['train', 'val', 'test']
        self.out_data_dir = Path('save/processed/TextVQA')
        self.max_nums = [-1, -1, -1]

    def create_data(self, ):
        for sid, split in enumerate(self.splits):
            filename = self.data_dir[split]
            img_path = self.img_dir[split]
            if not os.path.exists(self.out_data_dir):
                os.makedirs(self.out_data_dir)
            save_path = os.path.join(self.out_data_dir, f"{split}.jsonl")

            with open(filename, "r", encoding="utf-8") as fp:
                data = json.load(fp)['data']
            all_results = {}
            drop_num, tot_item_num = 0, 0
            for c_data in tqdm.tqdm(data):
                if split == "test":
                    answer = ""
                    answer_list = []
                else:
                    counts = Counter(c_data["answers"])
                    answer = sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]
                    answer_list = c_data["answers"]
                image_path = os.path.join(img_path, c_data["image_id"] + ".jpg")
                if not os.path.exists(image_path):
                    print(f"not found: {image_path}")
                    drop_num += 1
                c_data = {
                    "unique_id": "TextVQA-{}".format(c_data["question_id"]),
                    "question":c_data["question"],
                    "answer": answer,
                    "answer_list": answer_list,
                    # "image_path": image_path
                }
                all_results[image_path] = all_results.get(image_path, []) + [c_data]
                tot_item_num += 1
            
            max_count = self.max_nums[sid]
            with open(save_path, 'w') as f:
                item_num = 0
                for count, image_path in enumerate(all_results):
                    if count == max_count:
                        break
                    out_dict = {'image_path': image_path, 'metadata': all_results[image_path]}
                    f.write(json.dumps(out_dict) + '\n')
                    item_num += len(all_results[image_path])
            print(f"[TextVQA - {split}]: #images/#total = {count}/{len(all_results)}, #QAs/#total = {item_num}/{tot_item_num}")
    
                


if __name__ == '__main__':

    dataset = InstructData()
    dataset.create_data()