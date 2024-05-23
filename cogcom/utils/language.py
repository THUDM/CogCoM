import torch.nn.functional as F
from utils.com_utils import manipulate_image
from transformers import LlamaTokenizer
import re
import numpy as np
import torch



# def apply_rotary_pos_emb_index_bhs_q(q, cos, sin, position_id):
#     # batch_size, num_head, seq_len, hidden_size
#     cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(1), \
#                F.embedding(position_id, sin.squeeze(1)).unsqueeze(1)
#     q = (q * cos) + (rotate_half(q) * sin)
#     return q


def chat_history_to_prompt(self, history, query, eoi_tag="<img>"):
    """
      @history: str
    """
    prompt = ""
    if history:
        prompt = history
    img_pos = 0 if query.find(eoi_tag) < 0 else query.find(eoi_tag) # default position of image is at begining
    prompt = prompt + query[:img_pos] + '<EOI>' + query[img_pos:]
    return prompt


_history_to_prompt = {
    "base": chat_history_to_prompt,
    "chat": chat_history_to_prompt
}



def llama2_tokenizer(tokenizer_path, signal_type="base"):
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 32000
    tokenizer.boi = "[IMG]"
    tokenizer.eoi = "[/IMG]"
    assert signal_type in ["base", "chat"]
    tokenizer.signal_type= signal_type
    return tokenizer


def llama_args_postprocessing(self, kwargs):
    if 'vision_attention_mask' in kwargs and 'pre_image' in kwargs:
        pre_image = kwargs['pre_image']
        vision_attention_mask = kwargs['vision_attention_mask']
        lengths = vision_attention_mask.sum(dim=-1)
        for i, l in enumerate(lengths):
            l = int(l.item())
            kwargs['attention_mask'][i, :, :, pre_image+l:pre_image+self.image_length] = 0.
            kwargs['attention_mask'][i, :, pre_image+l:pre_image+self.image_length] = 0.
            kwargs['position_ids'][i, pre_image+l:] -= self.image_length - l
    return kwargs


class llama2_text_processor:
    def __init__(self, tokenizer, max_target_length=2048, image_length=1225, model=None):
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.image_length = image_length
        self.hist_max = 20000

    def __call__(self, caption, prompt="", history=None, eoi_tag="<img>"):
        """ The prompt using <img> to decide the position of image.
        """
        caption = caption if isinstance(caption, str) else str(caption)
        prompt = prompt if isinstance(prompt, str) else str(prompt)
        prompt = self.replace_tags_with_empty(prompt)
        caption = self.replace_tags_with_empty(caption)
        # history = []
        # prompt = self.history_to_prompt(history, prompt, add_eoi_first=True)
        # prompt = self.history_to_prompt(history, prompt)
        # prompt = self.history_to_prompt(history, prompt, eoi_tag=eoi_tag)
        prompt = self.history_to_prompt("", prompt, eoi_tag=eoi_tag)

        input_ids = [self.tokenizer.bos_token_id]

        prompt_splits = prompt.split('<EOI>')
        caption_splits = caption.split('<EOI>')
        if len(prompt_splits) > 0:
            input_ids.extend(self.tokenizer.encode(prompt_splits[0], add_special_tokens=False))
            pre_image = len(input_ids)
        for tokens in prompt_splits[1:]:
            tokens_with_img = [-100] + self.tokenizer.encode(tokens, add_special_tokens=False)
            input_ids.extend(tokens_with_img)
        context_length = len(input_ids) + (len(prompt_splits)-1) * (self.image_length + 1)
        if context_length > self.max_target_length - 50:
            return None
        if len(caption_splits) > 0:
            # input_ids.extend(self.tokenizer.encode(caption_splits[0], add_special_tokens=False)) # the longer caption (than the tokenizer defined) may cause error
            input_ids.extend(self.tokenizer.encode(caption_splits[0], add_special_tokens=False, truncation=True)) # the longer caption (than the tokenizer defined) may cause error
        for tokens in caption_splits[1:]:
            tokens_with_img = [-100] + self.tokenizer.encode(tokens, add_special_tokens=False)
            input_ids.extend(tokens_with_img)

        if len(input_ids) > self.max_target_length - self.image_length - 5:
            if(len(input_ids)) > self.max_target_length - self.image_length + 200:
                return None
            input_ids = input_ids[:self.max_target_length - self.image_length - 5]

        input_ids += [self.tokenizer.eos_token_id]

        while -100 in input_ids:
            img_idx = input_ids.index(-100)
            input_ids = input_ids[:img_idx] + [0] * (self.image_length + 1) + [-1] + input_ids[img_idx+1:]

        image_position = []
        while -1 in input_ids:
            img_idx = input_ids.index(-1)
            input_ids[img_idx] = 0
            image_position.append(img_idx)
        mem_len = torch.tensor(len(input_ids)).unsqueeze(0)

        # print(input_ids)
        image_embed_mask = [0] * len(input_ids)
        vision_expert_mask = [0] * len(input_ids)
        image_rope_mask = [0] * len(input_ids)
        for idx in image_position:
            image_embed_mask[idx-self.image_length-1: idx+1] = [1] * (self.image_length + 2)
            vision_expert_mask[idx-self.image_length-1: idx] = [1] * (self.image_length + 1)
            image_rope_mask[idx - self.image_length: idx] = [1] * self.image_length
        attention_mask = [1] * len(input_ids)
        labels = [-100] * context_length + input_ids[context_length:]

        pad_len = self.max_target_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        attention_mask = attention_mask + [1] * pad_len
        vision_expert_mask = vision_expert_mask + [0] * pad_len
        image_embed_mask = image_embed_mask + [0] * pad_len
        image_rope_mask = image_rope_mask + [0] * pad_len
        np_mask = np.tril(np.expand_dims(np.array(attention_mask), 0).repeat(len(attention_mask), 0))
        labels = labels + [-100] * pad_len

        for idx in image_position:
            labels[idx-self.image_length-1: idx+1] = [-100] * (self.image_length + 2)

        position_ids = []
        pid = -1
        for i in range(len(input_ids)):
            if image_rope_mask[i] == 0 or (i > 0 and image_rope_mask[i] != image_rope_mask[i - 1]):
                pid += 1
            position_ids.append(pid)

        input_ids = torch.tensor(input_ids).unsqueeze(0)
        labels = torch.tensor(labels).unsqueeze(0)
        attention_mask = torch.from_numpy(np_mask).unsqueeze(0).unsqueeze(0)
        image_embed_mask = torch.tensor(image_embed_mask).unsqueeze(0)
        vision_expert_mask = torch.tensor(vision_expert_mask).unsqueeze(0)
        image_rope_mask = torch.tensor(image_rope_mask).unsqueeze(0)
        position_ids = torch.tensor(position_ids).unsqueeze(0)
        pre_image = torch.tensor(pre_image).unsqueeze(0).long()
        context_length = torch.tensor(context_length).unsqueeze(0).long()

        mems_mask = torch.zeros(self.hist_max).unsqueeze(0)

        return {'input_ids': input_ids, 'labels': labels, 'position_ids': position_ids, 'attention_mask': attention_mask, 'image_embed_mask': image_embed_mask,
                'context_length': context_length, 'image_position': image_position, 'vision_expert_mask': vision_expert_mask, 'image_rope_mask': image_rope_mask,
                'pre_image': pre_image, 'mem_len': mem_len, 'mems_mask': mems_mask
                }

    # def history_to_prompt(self, history, query, add_eoi_first=False):
    #     # return _history_to_prompt(self, history, query, add_eoi_first)
    #     return _history_to_prompt[self.tokenizer.signal_type](self, history, query)

    def history_to_prompt(self, history, query, eoi_tag="<img>"):
        # return _history_to_prompt(self, history, query, add_eoi_first)
        return _history_to_prompt[self.tokenizer.signal_type](self, history, query, eoi_tag=eoi_tag)

    def pre_caption(self, caption):
        # caption = re.sub(
        #     r"([.!\"()*#:;~])",
        #     " ",
        #     caption,
        # )

        # caption = re.sub(
        #     r"([^\w\s,.?，。？、()（）]|_)+",
        #     " ",
        #     caption,
        # )

        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")
        return caption

    def replace_tags_with_empty(self, text):
        return re.sub('<pad>|<s>|</s>|<EOI>', '', text)

from functools import partial
def get_masks_and_position_ids(seq, image_logits_mask):
    tokens = seq.unsqueeze(0)

    attention_mask = torch.ones((1, len(seq), len(seq)), device=tokens.device)
    attention_mask.tril_()
    attention_mask.unsqueeze_(1)

    position_ids = []
    pid = -1
    for i in range(len(image_logits_mask[0])):
        if image_logits_mask[0][i] == 0 or (i > 0 and image_logits_mask[0][i] != image_logits_mask[0][i - 1]):
            pid += 1
        position_ids.append(pid)
    for i in range(tokens.shape[1]-image_logits_mask.shape[1]):
        pid += 1
        position_ids.append(pid)
    position_ids = torch.tensor(position_ids, dtype=torch.long, device=tokens.device)
    position_ids = position_ids.unsqueeze(0)
    # breakpoint()
    return tokens, attention_mask, position_ids

class llama2_text_processor_inference:
    def __init__(self, tokenizer, max_target_length=823, image_length=1225, model=None, no_prompt=False, english=True):
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.image_length = image_length
        self.sep = "<unk>"
        self.invalid_slices = []
        self.no_eoi = True
        self.hist_max = 20000

    # def __call__(self, prompt="", history=None):
    def __call__(self, prompt="", history=None, hard_prompt=False, eoi_tag='<img>'):
        prompt = prompt if isinstance(prompt, str) else str(prompt)
        prompt = self.replace_tags_with_empty(prompt)
        # history = []
        # prompt = self.history_to_prompt(history, prompt, add_eoi_first=True)
        # prompt = self.history_to_prompt(history, prompt, add_eoi_first=True)
        prompt = self.history_to_prompt("", prompt, eoi_tag=eoi_tag)

        input_ids = [self.tokenizer.bos_token_id]

        prompt_splits = prompt.split('<EOI>')
        if len(prompt_splits) > 0:
            input_ids.extend(self.tokenizer.encode(prompt_splits[0], add_special_tokens=False))
            pre_image = len(input_ids)
        for tokens in prompt_splits[1:]:
            tokens_with_img = [-100] + self.tokenizer.encode(tokens, add_special_tokens=False)
            input_ids.extend(tokens_with_img)

        # if len(input_ids) > self.max_target_length - self.image_length - 5:
        #     if(len(input_ids)) > self.max_target_length - self.image_length + 200:
        #         raise Exception("token too long")
        #     input_ids = input_ids[:self.max_target_length - self.image_length - 5]

        while -100 in input_ids:
            img_idx = input_ids.index(-100)
            input_ids = input_ids[:img_idx] + [0] * (self.image_length + 1) + [-1] + input_ids[img_idx + 1:]
        mem_len = torch.tensor(len(input_ids)).unsqueeze(0)

        image_position = []
        while -1 in input_ids:
            img_idx = input_ids.index(-1)
            input_ids[img_idx] = 0
            image_position.append(img_idx)

        # print(input_ids)
        image_embed_mask = [0] * len(input_ids)
        vision_expert_mask = [0] * len(input_ids)
        image_rope_mask = [0] * len(input_ids)
        for idx in image_position:
            image_embed_mask[idx - self.image_length - 1: idx + 1] = [1] * (self.image_length + 2)
            vision_expert_mask[idx - self.image_length - 1: idx] = [1] * (self.image_length + 1)
            image_rope_mask[idx - self.image_length: idx] = [1] * self.image_length

        input_ids = torch.tensor(input_ids).unsqueeze(0)
        image_embed_mask = torch.tensor(image_embed_mask).unsqueeze(0)
        vision_expert_mask = torch.tensor(vision_expert_mask).unsqueeze(0)
        image_rope_mask = torch.tensor(image_rope_mask).unsqueeze(0)
        pre_image = torch.tensor(pre_image).unsqueeze(0).long()

        mems_mask = torch.zeros(self.hist_max).unsqueeze(0)
        return {'input_ids': input_ids, 'image_embed_mask': image_embed_mask, 'vision_expert_mask': vision_expert_mask,
                'image_rope_mask': image_rope_mask, 'pre_image': pre_image, 'mem_len': mem_len, 'mems_mask': mems_mask}

    # def history_to_prompt(self, history, query, add_eoi_first=False):
    #     # return _history_to_prompt(self, history, query, add_eoi_first)
    #     return _history_to_prompt[self.tokenizer.signal_type](self, history, query, add_eoi_first)
    
    def history_to_prompt(self, history, query, eoi_tag="<img>"):
        # return _history_to_prompt(self, history, query, add_eoi_first)
        return _history_to_prompt[self.tokenizer.signal_type](self, history, query, eoi_tag=eoi_tag)

    def pre_caption(self, caption):
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")
        return caption

    def replace_tags_with_empty(self, text):
        return re.sub('<pad>|<s>|</s>|<EOI>', '', text)

    # def process_response(self, response):
    #     return response.rstrip('</s>')

    def process_response(self, response, img):
        done = True
        new_img = img
        response =  response.rstrip('</s>')
        query = ""

        try:
            # ptr = re.compile(r'([A-Z_]+)\((.*?)\)')
            ptr = re.compile(r'([A-Za-z_]+)\((.*?)\)')
            matches = ptr.findall(response)
        except Exception as e:
            print(e)
        if matches:
            func, param = matches[-1] # acquire the last manipulation
            if 'crop_and_zoomin' in func.lower():
                try:
                    onbox = eval(re.match(r'.*?(\[[\[\]\d,\s]+\]).*', param).group(1))
                except:
                    onbox = None
                new_img = manipulate_image(img, func, onbox)
                done = False
            else:
                new_img = manipulate_image(img, func, param)
            # query = self.history_to_prompt(history=True, query='', add_eoi_first=False)
            query = self.history_to_prompt(history="", query='')
        return response, query, new_img, done
    
    def get_func(self, inputs, **kwargs):
        get_func = partial(get_masks_and_position_ids, image_logits_mask=kwargs['image_rope_mask'])
        return get_func


if __name__ == '__main__':
    
    from sat.tokenization import get_tokenizer
    tokenizer = get_tokenizer(outer_tokenizer=llama2_tokenizer('/share/official_pretrains/hf_home/vicuna-7b-v1.5/'))

    text_processor = llama2_text_processor(tokenizer, 2048, 1225)
    ret = text_processor('This is caption.', 'This is prompt.')
    print(ret)