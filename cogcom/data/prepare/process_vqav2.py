"""answer question base on the content of the image
@File    :   process_vqav2.py
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

class InstructData:

    def __init__(self, ) -> None:
        self.data_dir = 'save/raw/VQA_V2'
        self.img_dir = {
            'train': 'save/raw/MSCOCO/MSCOCO2014/train2014',
            'val': 'save/raw/MSCOCO/MSCOCO2014/val2014'
        }
        self.splits = ['train', 'val']
        self.out_data_dir = Path('save/processed/VQAv2')
        self.max_nums = [-1, -1]
        self.coco = {
            'train': COCO('save/raw/MSCOCO/MSCOCO2014/annotations/instances_train2014.json'),
            'val': COCO('save/raw/MSCOCO/MSCOCO2014/annotations/instances_val2014.json')
        }


    def create_data(self,):
        """ Process vqav2 datset and save into the out_data_dir for training and validating datasets.
        """
        for i, split in enumerate(self.splits):

            questions = json.load(open(os.path.join(self.data_dir,f'v2_OpenEnded_mscoco_{split}2014_questions.json'),'r'))
            questions = sorted(questions['questions'], key=lambda x:x['question_id'])
            answers = json.load(open(os.path.join(self.data_dir,f'v2_mscoco_{split}2014_annotations.json'),'r'))
            answers = sorted(answers['annotations'], key=lambda x:x['question_id'])

            all_samples = {}
            img2ann = {}
            tot_item_num = 0
            for q, a in tqdm.tqdm(list(zip(questions, answers)), desc='VQAv2'):
                assert a['question_id'] == q['question_id']
                image_path = os.path.join(self.img_dir[split], f"COCO_{split}2014_{q['image_id']:012d}.jpg")
                assert os.path.exists(image_path)
                out_dict = {
                            'unique_id': 'VQAv2-'+str(q['question_id']),
                            'image_source': 'coco2014',
                            'question': q['question'],
                            'answer': a['multiple_choice_answer'],
                            'answer_list': [ans["answer"] for ans in a["answers"]],
                            # 'image_path': image_path
                }
                all_samples[image_path] = all_samples.get(image_path, []) + [out_dict]
                tot_item_num += 1

                if image_path not in img2ann:
                    anns = self.coco[split].loadAnns(self.coco[split].getAnnIds([q['image_id']]))
                    # new_anns = {'bboxes':[], 'cats': []}
                    new_anns = []
                    for a in anns:
                        new_anns.append(
                            {'cat': self.coco[split].loadCats([a['category_id']])[0]['name'], 'supercat': self.coco[split].loadCats([a['category_id']])[0]['supercategory'], 'bbox': a['bbox']}
                        )
                        # new_anns['cats'].append(self.coco[split].loadCats([a['category_id']]))
                        # new_anns['bboxes'].append(a['bbox'])
                    img2ann[image_path] = new_anns
                    

            os.makedirs(self.out_data_dir, exist_ok=True)
            save_path = self.out_data_dir / f'{split}.jsonl'
            max_count = self.max_nums[i]
            with open(save_path, 'w') as f:
                item_num = 0
                for i, image_path in enumerate(all_samples):
                    if i == max_count:
                        break
                    out_dict = {'image_path': image_path, 'metadata': all_samples[image_path], 'image_ann': img2ann[image_path]}
                    f.write(json.dumps(out_dict) + '\n')
                    item_num += len(all_samples[image_path])
            print(f'[VQAv2 - {split}]: #images/#total = {i}/{len(all_samples)}, #QAs/#total = {item_num}/{tot_item_num}')

                


if __name__ == '__main__':

    dataset = InstructData()
    dataset.create_data()