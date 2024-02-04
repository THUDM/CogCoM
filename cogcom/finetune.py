import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import numpy as np
import copy
from braceexpand import braceexpand
from sat import mpu, get_args, get_tokenizer
from sat.training.deepspeed_training import training_main
from sat.helpers import print_rank0
from collections import defaultdict
from models import FineTuneTrainCogCoMModel
from utils import llama2_text_processor, llama2_text_processor_inference, llama2_tokenizer, get_image_processor
from utils import CoMDatasetProcessor, InstructDatasetProcessor


def disable_untrainable_params(self):
    enable =  ["linear_proj", 'mixins.mlp.vision_gate_proj', 'mixins.mlp.vision_dense_h_to_4h_list', 'mixins.mlp.vision_dense_4h_to_h_list', 
               'mixins.rotary.vision_query_key_value_list', 'mixins.rotary.vision_dense_list']
    total_trainable = 0
    for n, p in self.named_parameters():
        flag = False
        for e in enable:
            if type(e) is tuple and e[0].lower() in n.lower() and e[1].lower() in n.lower():
                flag = True
                break
            elif type(e) is not tuple and e.lower() in n.lower():
                flag = True
                break
        if not flag:
            p.requires_grad_(False)
        else:
            print_rank0(n)
            total_trainable += p.numel()
    print_rank0("***** Total trainable parameters: "+str(total_trainable)+" *****")

FineTuneTrainCogCoMModel.disable_untrainable_params = disable_untrainable_params




def data_collator_com(chains):
    """ Concatenate turns with same indices into a batch.
      Args:
        @chains: A batch of chains, where each one is a multi-turn chain.
      Return:
        {
          0: { // 1st turn
            'input_ids': 'a batch of input_ids',
            'vision_image': 'a batch of images',
          },
          1: {  // 2nd turn
            'input_ids': 'a batch of input_ids',
            'vision_image': 'a batch of images',
          },
          ...
          'chain_mask': [[1,1,1..,0], [1,..,0], * b] # mark len of each chain
        }
    """
    def to_tensor(data):
        if isinstance(data, list):
            data = torch.tensor(data)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return data

    def merge(ori_turn, new_turn=None):
        # create an empty counterpart if the new_turn is None.
        for k in ori_turn:
            ori_turn[k] = to_tensor(ori_turn[k])
            if new_turn is None:
                data = copy.copy(ori_turn[k])
            else:
                data = to_tensor(new_turn[k])
            if isinstance(ori_turn[k], torch.Tensor):
                ori_turn[k] = torch.cat([ori_turn[k], data], dim=0)
            else:
                if not isinstance(ori_turn[k], list):
                    ori_turn[k] = [ori_turn[k]] + [data]
                else:
                    ori_turn[k].append(data)
        return ori_turn

    chains = list(chains)
    srt_chains = sorted(chains, key=lambda x:len(x), reverse=True)
    final_chains = {k:v for k,v in enumerate(srt_chains[0])}
    final_chains['chain_mask'] = [[1]*len(srt_chains[0])]

    max_turns = len(srt_chains[0])
    for i, chain in enumerate(srt_chains[1:]):
        final_chains['chain_mask'].append([])
        for turnid in range(max_turns):
            turn_data = None
            mask = 0
            if turnid < len(chain):
                turn_data = chain[turnid]
                mask = 1
            final_chains[turnid] = merge(final_chains[turnid], turn_data) # concatenate all key-values
            final_chains['chain_mask'][i+1].append(mask)
    final_chains['chain_mask'] = torch.Tensor(final_chains['chain_mask']).T
    return final_chains

    


def broadcast_auto_com(chains):
    type2list = defaultdict(list)
    other = []
    for k in chains[0]:
        if type(chains[0][k]) is torch.Tensor:
            type2list[chains[0][k].dtype].append(k)
        else:
            other.append(k)
    new_chains = {}
    for turnid in chains:
        if isinstance(turnid, int):
            new_chains[turnid] = {}
            for k in type2list:
                new_chains[turnid].update(mpu.broadcast_data(type2list[k], chains[turnid], k))
            for k in other:
                new_chains[turnid][k] = chains[turnid][k]
    new_chains['chain_mask'] = mpu.broadcast_data(['chain_mask'], chains, chains['chain_mask'].dtype)['chain_mask']

    return new_chains

def get_batch_com(data_iterator, args, timers):
    def to_fp_bf(data_dict):
        for k in data_dict:
            if type(data_dict[k]) is torch.Tensor and data_dict[k].dtype is not torch.int32 and data_dict[k].dtype is not torch.long:
                if args.fp16:
                    data_dict[k] = data_dict[k].half()
                elif args.bf16:
                    data_dict[k] = data_dict[k].bfloat16()
        return data_dict
    
    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        chains = next(data_iterator)
    else:
        chains = None
    timers('data loader').stop()
    try:
        chains_b = broadcast_auto_com(chains)
    except Exception as e:
        print(f"{e}: cause exception on broadcast!")
        assert 1==2
    for k in chains_b:
        if isinstance(k, int):
            chains_b[k] = to_fp_bf(chains_b[k])
        else:
            chains_b[k] = to_fp_bf({'_': chains_b[k]})['_']
    return chains_b


import einops
from torch.nn import CrossEntropyLoss
from sat.generation.autoregressive_sampling import update_mems
from models.com_memory import update_mems
def forward_step_com(data_iterator, model, args, timers):
    """ Multi-turn: when there is a resulted **new image**, the new turn will be started to input the new image.
          Version-1: Using the Memory for simplicity.
    """
    # Get the batch.
    timers('batch generator').start()
    chains_b = get_batch_com( # each sample is a chain, including new image
            data_iterator, args, timers)
    timers('batch generator').stop()

    history = None
    tot_loss = 0
    turnid = 0
    mems, mems_mask = None, None
    while turnid in chains_b:
        turn_b = chains_b[turnid]
        turn_labels = turn_b.pop('labels')
        turn_mask = chains_b['chain_mask'][turnid]
        turn_mask = turn_mask.to(torch.float32)
        turn_mem_lens = turn_b.get('mem_len')
        if mems_mask is not None:
            turn_b['mems_mask'][:, :mems_mask.shape[1]] = mems_mask

        logits, *output_per_layers = model(**turn_b, mems=mems)
        lm_logits = logits.to(torch.float32)

        mem_kv = [o['mem_kv'] for o in output_per_layers] # list (num_layers) of [batch, query_length, 2d]
        mems, mems_mask = update_mems(mem_kv, mems, mems_mask, turn_mem_lens, max_memory_length=100000) # [num_layers, batch, memory_length, 2d]

        # Shift so that tokens < n predict n
        shift_labels = turn_labels[..., 1:].contiguous()
        shift_logits = lm_logits[..., -1-shift_labels.size(-1):-1, :].contiguous()

        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        mask = (shift_labels!=-100).view(-1)
        turn_mask = einops.repeat(turn_mask, 'i -> (i s)', s=shift_labels.size(1)) # mask empty chain
        loss = loss[mask] * turn_mask[mask]
        loss = loss.mean()
        loss = loss.to(torch.float32)
        tot_loss += loss

        turnid += 1

    avg_loss = tot_loss / len(chains_b)
    return avg_loss, {'loss': tot_loss}

import random
def get_tar_files(path, suffix='tar'):
    pathes = path.split(',')
    tar_files = []
    for p in pathes:
        local_files = []
        if '*' in p:
            include_dirs, nr = p.split('*')
            if '#' in nr:
                n, r = nr.split('#')
            else:
                n = nr
                r = -1
        else:
            include_dirs = p
            n = 1
            r = -1
        repeat_nums, remain_nums = int(n), int(r)
            
        if include_dirs.endswith(suffix): # path/to/name-{000000..000024}.tar
            # tar_files.extend(list(braceexpand(include_dirs)) * repeat_nums)
            local_files.extend(list(braceexpand(include_dirs)) * repeat_nums)
        else: # path/to/dataset_name
            for cur_dir, _, files in os.walk(include_dirs, followlinks=True):
                for f in files:
                    # if f.endswith('.tar') and os.path.getsize(os.path.join(cur_dir,f)) > 0:
                    if f.endswith(suffix) and os.path.getsize(os.path.join(cur_dir,f)) > 0:
                        # tar_files.extend([os.path.join(cur_dir,f)]*repeat_nums)
                        local_files.extend([os.path.join(cur_dir,f)]*repeat_nums)
        if remain_nums > 0: # remain samples
            random.shuffle(local_files)
            local_files = local_files[:remain_nums]
        tar_files.extend(local_files)
    print_rank0(f'find {len(tar_files)} tars in all...')
    return tar_files

def create_dataset_function(data_processors, image_processor, text_processor, path, args):
    path, proc_name = path
    urls = get_tar_files(path, 'tar')
    print_rank0(f'[Using data path is]: {path}')
    dataset = data_processors[proc_name](urls, args, image_processor, text_processor)
    return dataset


from functools import partial
from sat.model.finetune.lora2 import LoraMixin
from sat.model.finetune.prompt_tuning import PTuningV2Mixin
if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    py_parser.add_argument('--max_source_length', type=int)
    py_parser.add_argument('--max_target_length', type=int)
    py_parser.add_argument('--version', type=str, default='cogcom_base', help="version of the model you want to load")
    py_parser.add_argument('--ignore_pad_token_for_loss', type=bool, default=True)
    py_parser.add_argument('--from_pretrained', default='', type=str, help="pretrained model you want to load")
    py_parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    py_parser.add_argument("--vit_checkpoint_activations", action='store_true')
    py_parser = FineTuneTrainCogCoMModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    if args.use_qlora:
        args.device = 'cpu'

    model, args = FineTuneTrainCogCoMModel.from_pretrained(args.from_pretrained, args, overwrite_args={'model_parallel_size': args.model_parallel_size} if args.model_parallel_size != 1 else {})
    
    if args.use_ptuning:
        model.add_mixin("ptuning", PTuningV2Mixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.pre_seq_len))
    if args.use_lora:
        model.add_mixin("lora", LoraMixin(args.num_layers, args.lora_rank, layer_range=args.layer_range), reinit=True)
        model.get_mixin("eva").vit_model.add_mixin("lora", LoraMixin(args.eva_args['num_layers'], args.lora_rank, layer_range=args.layer_range), reinit=True)
    elif args.use_qlora:
        model.add_mixin("lora", LoraMixin(args.num_layers, args.lora_rank, layer_range=args.layer_range, qlora=True), reinit=True)
        
    if args.use_qlora and torch.cuda.is_available():
        model = model.to('cuda')

    data_processors = {
        'CoM': CoMDatasetProcessor,
        'Instruct': InstructDatasetProcessor
    }
    args.train_data = [(path.split('#')[0], path.split('#')[1]) for path in args.train_data] # path: Dir#Processor
    
    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type="chat")
    image_processor = get_image_processor(490)
    text_processor = llama2_text_processor(tokenizer, args.max_source_length+args.max_target_length, image_length=1225, model=model)

    model = training_main(args, model_cls=model, forward_step_function=forward_step_com, create_dataset_function=partial(create_dataset_function, data_processors, image_processor, text_processor), collate_fn=data_collator_com)
    if args.use_lora:
        model.get_mixin("lora").merge_lora()
        model.get_mixin("eva").vit_model.get_mixin("lora").merge_lora()
        args.use_lora = False
        args.save = "checkpoints/merged_lora_cogcom{}".format(args.eva_args["image_size"][0])
        from sat.training.model_io import save_checkpoint
        save_checkpoint(1, model, None, None, args)