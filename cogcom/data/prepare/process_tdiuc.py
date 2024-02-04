"""answer question base on the content of the image
@File    :   process_tdiuc.py
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
        self.data_dir = 'save/raw/TDIUC/'
        self.img_dir = 'save/raw/MSCOCO/MSCOCO2014'
        self.splits = ['train', 'val']
        self.out_data_dir = Path('save/processed/TDIUC')
        self.max_nums = [-1, -1]


    def create_data(self, ) -> None:
        """
        """
        def select_answer_by_confidence(answers):
            confidenced_answers = [answer["answer"] for answer in answers if answer["answer_confidence"] == "yes"]
            if len(confidenced_answers) == 0:
                return None
            counts = Counter(confidenced_answers)
            counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            return counts[0][0]

        for sid, split in enumerate(self.splits):
        
            question_file = os.path.join(self.data_dir, f'Questions/OpenEnded_mscoco_{split}2014_questions.json')
            annotaion_file = os.path.join(self.data_dir, f'Annotations/mscoco_{split}2014_annotations.json')
            if not os.path.exists(self.out_data_dir):
                os.makedirs(self.out_data_dir)
            save_path = os.path.join(self.out_data_dir, f"{split}.jsonl")
            with open(question_file) as f1, open(annotaion_file) as f2:
                questions, anns = json.load(f1), json.load(f2)
                
            sorted_qs = sorted(questions['questions'], key=lambda x:x['question_id'])
            sorted_anns = sorted(anns['annotations'], key=lambda x:x['question_id'])
            all_data, drop_num, tot_item_num = {}, 0, 0
            for idx in tqdm.tqdm(range(len(sorted_qs)), desc='TDIUC'):
                q_info, a_info = sorted_qs[idx], sorted_anns[idx]
                assert q_info["image_id"] == a_info["image_id"]
                qa_info = q_info | a_info
                img_path = os.path.join(self.img_dir, f'{split}2014', f"COCO_{split}2014_{qa_info['image_id']:012d}.jpg")
                if not os.path.exists(img_path):
                    drop_num += 1
                    # print(f'not found {img_path}.')
                    continue
                answer = select_answer_by_confidence(qa_info["answers"])
                if answer is None:
                    drop_num += 1
                    print(f'no confidenced answer!')
                    continue
                c_data = {
                    "unique_id": "TDIUC-%s" % qa_info["question_id"],
                    "question": qa_info["question"],
                    "answer": answer,
                    "question_type": qa_info["question_type"],
                    "ans_source": qa_info["ans_source"],
                    # 'image_path': img_path
                }

                all_data[img_path] = all_data.get(img_path, []) + [c_data]

                tot_item_num += 1

            with open(save_path, 'w') as f:
                max_count = self.max_nums[sid]
                item_num = 0
                count = 0
                for count, image_path in enumerate(all_data):
                    if count == max_count:
                        break
                    out_dict = {'image_path': image_path, 'metadata': all_data[image_path]}
                    f.write(json.dumps(out_dict) + '\n')
                    item_num += len(all_data[image_path])
            print(f"[TDIUC - {split}]: #images/#total = {count}/{len(all_data)}, #QAs/#total={item_num}/{tot_item_num}")
    
                


if __name__ == '__main__':

    dataset = InstructData()
    dataset.create_data()