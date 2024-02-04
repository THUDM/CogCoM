import os
import random
import argparse
import json
import math
import datetime
import glob
from multiprocessing import Process
from webdataset import TarWriter


def make_wds_shards(pattern, num_sample_per_shard, num_workers, samples, map_func, **kwargs):
    # random.shuffle(samples)

    num_shards = math.ceil(len(samples) / num_sample_per_shard)

    samples_per_shards = [samples[i::num_shards] for i in range(num_shards)]
    shard_ids = list(range(num_shards))

    processes = [
        Process(
            target=write_partial_samples,
            args=(
                pattern,
                shard_ids[i::num_workers],
                samples_per_shards[i::num_workers],
                map_func,
                kwargs
            )
        )
        for i in range(num_workers)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()



def write_partial_samples(pattern, shard_ids, samples, map_func, kwargs):
    i = 0
    for shard_id, samples in zip(shard_ids, samples):
        unique_id = f'{shard_id}_{i}'
        write_samples_into_single_shard(pattern, shard_id, samples, map_func, kwargs, unique_id)
        i += 1


def write_samples_into_single_shard(pattern, shard_id, samples, map_func, kwargs, unique_id):
    fname = pattern % shard_id
    print(f"[{datetime.datetime.now()}] start to write samples to shard {fname}")
    stream = TarWriter(fname, **kwargs)
    size, i = 0, 0
    for item in samples:
        # size += stream.write(map_func(item, uid=f'{unique_id}_{i}'))
        ex = map_func(item, uid=f'{unique_id}_{i}')
        if ex:
            size += stream.write(ex)
            i += 1
    stream.close()
    print(f"[{datetime.datetime.now()}] complete to write samples to shard {fname}")
    return size



def map_func(item, uid):
    im_path = item.pop('image_path')
    with open(im_path, "rb") as stream:
        image = stream.read()
    # sample = {
    #     "__key__": uid + '_' + os.path.splitext(os.path.basename(im_path))[0],
    #     "jpg": image,
    #     'prompt': item.pop('prompt'),
    #     'txt': item.pop('target_txt'),
    #     'task_name': item.pop('task_name'),
    #     'unique_id': item.pop('unique_id')
    # }
    metadata = []
    for qa in item['metadata']:
        if qa.get('com_founds', []): # must have at least one viable path
            metadata.append(qa)

    sample = None
    if len(metadata) > 0:
        sample = {
            "__key__": uid,
            # "__key__": uid + '_' + os.path.splitext(os.path.basename(im_path))[0],
            "jpg": image,
            'metadata.pyd': metadata
        }
    
    return sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default=None, type=str)
    parser.add_argument('--in_dir', default=None, type=str)
    parser.add_argument('--out_dir', default=None, type=str)
    args = parser.parse_args()

    data_dir = args.in_dir
    save_dir = args.out_dir
    data_name = args.data_name
    os.makedirs(save_dir, exist_ok=True)

    if data_name is None:
        data_name = os.path.basename(data_dir)


    # Process all datasets
    train_results = []
    # train_files = list(glob.glob('training_data/*/*',recursive=True))
    train_files = list(glob.glob(f'{data_dir}/*',recursive=True))
    train_lines = []
    for file_name in train_files:
            assert '.jsonl' in file_name
            # if  not 'train.jsonl' in file_name:
            #     continue
            with open(file_name,'r') as fin:
                for line in fin:
                    line = json.loads(line)
                    train_lines.append(line)

    print(f"In total {len(train_lines)} lines are loaded.")
    make_wds_shards(
        pattern=os.path.join(save_dir, f"com-{data_name}-%06d.tar"),
        num_sample_per_shard=100,
        num_workers=20,
        samples=train_lines,
        map_func=map_func,
    )




        