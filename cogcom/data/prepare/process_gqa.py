"""answer question base on the content of the image
@File    :   process_gqa.py
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
import glob

class InstructData:

    def __init__(self, ) -> None:
        self.data_dir = {'train': glob.glob('save/raw/GQA/train_all_questions/train_all_questions_**.json'),
                             'val':['save/raw/GQA/val_all_questions.json'],
                             'test': ['save/raw/GQA/testdev_balanced_questions.json']}
        self.img_dir = 'save/raw/GQA/images/'
        self.out_data_dir = Path('save/processed/GQA')
        self.splits = ['train', 'val', 'test']
        self.max_nums = [-1, -1, -1]
        

    def create_data(self, ):
        for sid, split in enumerate(self.splits):
            all_results , img2ann= {}, {}
            drop_num , tot_item_num = 0, 0
            for filename in self.data_dir[split]:
                with open(filename, "r", encoding='utf-8') as f:
                    processed_datas = list(json.load(f).values())
                for data in tqdm.tqdm(processed_datas, desc='GQA'):
                    if 'imageId' not in data or 'question' not in data or 'answer' not in data  or 'fullAnswer' not in data:
                        drop_num += 1
                        continue
                    image_path = os.path.join(self.img_dir, data['imageId'] + ".jpg")
                    if not os.path.exists(image_path):
                        # print(f'not found: {image_path}')
                        drop_num += 1
                        continue
                    prompt, txt, full_answer = data['question'], data['answer'], data['fullAnswer']
                    conversation = {
                        "unique_id": "GQA-{}-{}".format(split, "%09d" %(tot_item_num)),
                        "question":prompt,
                        "answer": txt,
                        "full_answer": full_answer,
                        # "image_path": image_path
                    }
                    tot_item_num += 1
                    all_results[image_path] = all_results.get(image_path, []) + [conversation]

            os.makedirs(self.out_data_dir, exist_ok=True)
            save_path = os.path.join(self.out_data_dir, f"{split}.jsonl")

            item_num = 0
            with open(save_path, 'w') as f:
                max_count = self.max_nums[sid]
                count = 0
                for count, image_path in enumerate(all_results):
                    if count == max_count:
                        break
                    out_dict = {'image_path': image_path, 'metadata': all_results[image_path]}
                    f.write(json.dumps(out_dict) + '\n')
                    item_num += len(all_results[image_path])
            print(f"[GQA - {split}]: #images/#total = {count}/{len(all_results)}, #QAs/#total={item_num}/{tot_item_num} {drop_num} images are not found.")
    
                


if __name__ == '__main__':

    dataset = InstructData()
    dataset.create_data()