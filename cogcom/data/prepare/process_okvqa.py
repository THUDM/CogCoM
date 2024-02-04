"""answer question base on the content of the image
@File    :   process_okvqa.py
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
from pycocotools.coco import COCO
from collections import Counter

class InstructData:

    def __init__(self, ) -> None:
        self.data_dir = 'save/raw/OK-VQA/'
        self.img_dir = {
            'train': 'save/raw/MSCOCO/MSCOCO2014/train2014',
            'val': 'save/raw/MSCOCO/MSCOCO2014/val2014'
        }
        self.splits = ['train', 'val']
        self.out_data_dir = Path('save/processed/OK-VQA')
        self.max_nums = [-1, -1]
        self.coco = {
            'train': COCO('save/raw/MSCOCO/MSCOCO2014/annotations/instances_train2014.json'),
            'val': COCO('save/raw/MSCOCO/MSCOCO2014/annotations/instances_val2014.json')
        }

    def create_data(self, ):
        def select_answer_by_confidence(answers):
            answer_list = [answer["answer"] for answer in answers]
            if len(answer_list) == 0:
                return None, None
            counts = Counter(answer_list)
            counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            return counts[0][0], answer_list

        for sid, split in enumerate(self.splits):
            question_file = os.path.join(self.data_dir, split, f'OpenEnded_mscoco_{split}2014_questions.json')
            annotaion_file = os.path.join(self.data_dir, split, f'mscoco_{split}2014_annotations.json')
            with open(question_file) as f1, open(annotaion_file) as f2:
                questions, anns = json.load(f1), json.load(f2)
                
            sorted_qs = sorted(questions['questions'], key=lambda x:x['question_id'])
            sorted_anns = sorted(anns['annotations'], key=lambda x:x['question_id'])
            all_data, img2ann, drop_num, tot_item_num = {}, {}, 0, 0
            for idx in tqdm.tqdm(range(len(sorted_qs)), desc='OKVQA'):
                q_info, a_info = sorted_qs[idx], sorted_anns[idx]
                assert q_info["image_id"] == a_info["image_id"]
                qa_info = q_info | a_info
                img_path = os.path.join(self.img_dir[split],f"COCO_{split}2014_{qa_info['image_id']:012d}.jpg")
                if not os.path.exists(img_path):
                    drop_num += 1
                    print(f'not found {img_path}.')
                    continue
                answer, answer_list = select_answer_by_confidence(qa_info["answers"])
                if answer is None:
                    drop_num += 1
                    print(f'no confidenced answer!')
                    continue
                c_data = {
                "unique_id": "OKVQA-%s" % qa_info["question_id"],
                "question": qa_info["question"],
                "answer": answer,
                "answer_list": answer_list,
                "question_type": anns["question_types"],
                # "image_path": img_path,
                }
                
                all_data[img_path] = all_data.get(img_path, []) + [c_data]

                if img_path not in img2ann:
                    cocoanns = self.coco[split].loadAnns(self.coco[split].getAnnIds([qa_info['image_id']]))
                    # new_anns = {'bboxes':[], 'cats': []}
                    new_anns = []
                    for a in cocoanns:
                        new_anns.append(
                            {'cat': self.coco[split].loadCats([a['category_id']])[0]['name'], 'supercat': self.coco[split].loadCats([a['category_id']])[0]['supercategory'], 'bbox': a['bbox']}
                        )
                        # new_anns['cats'].append(self.coco[split].loadCats([a['category_id']]))
                        # new_anns['bboxes'].append(a['bbox'])
                    img2ann[img_path] = new_anns
                tot_item_num += 1

            os.makedirs(self.out_data_dir, exist_ok=True)
            save_path = os.path.join(self.out_data_dir, f"{split}.jsonl")
            item_num = 0
            with open(save_path, 'w') as f:
                max_count = self.max_nums[sid]
                for count, image_path in enumerate(all_data):
                    if count == max_count:
                        break
                    out_dict = {'image_path': image_path, 'metadata': all_data[image_path], 'image_ann': img2ann[image_path]}
                    f.write(json.dumps(out_dict) + '\n')
                    item_num += len(all_data[image_path])
            print(f"[OKVQA - {split}]: #images/#total = {count}/{len(all_data)}, #QAs/#total = {item_num}/{tot_item_num}")
    
                


if __name__ == '__main__':

    dataset = InstructData()
    dataset.create_data()