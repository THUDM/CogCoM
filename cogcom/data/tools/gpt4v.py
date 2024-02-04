import re
import time
import random
import json
import glob
import openai
import requests
import traceback
import uuid
import argparse
from PIL import Image
import random
import json
import time

GPT4VTOKENS = [
    ""
]


def urlfy(img):
    files = {"c": open(img, "rb"), "e": 3600}
    response = requests.post("https://pb.pka.moe/", files=files)
    return response.json()["suggestUrl"]

def put_file(password, file_name, file_data):

    headers = {
        'accept': '*/*',
        'accept-language': 'en-US',
        'authorization': f'Bearer {password}',
        'content-type': 'application/json',
        'dnt': '1',
        'origin': 'https://mirror.llm.beauty',
        'referer': 'https://mirror.llm.beauty/?model=gpt-4',
        'sec-ch-ua': '"Not=A?Brand";v="99", "Chromium";v="118"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
    }

    json_data = {
        'file_name': file_name,
        'file_size': len(file_data),
        'use_case': 'multimodal',
    }

    response = requests.post('https://mirror.llm.beauty/backend-api/files', headers=headers, json=json_data)
    data = response.json()
    upload_url = data["upload_url"]
    file_id = data["file_id"]
    upload_headers = {
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Content-Type': 'application/pdf',
        'DNT': '1',
        'Origin': 'https://chat.openai.com',
        'Pragma': 'no-cache',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'cross-site',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Not)A;Brand";v="24", "Chromium";v="116"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'x-ms-blob-type': 'BlockBlob',
        'x-ms-version': '2020-04-08',
    }

    response = requests.put(upload_url, data=file_data, headers=upload_headers)
#    print(response.status_code)
    response = requests.post(
        f'https://mirror.llm.beauty/backend-api/files/{file_id}/uploaded',
        headers=headers,
        verify=False,
    )
#    print(response.text)
    return file_id



def ask_code(password, content = {}, metadata = {}):
    json_data = {
        'action': 'next',
        'messages': [
            {
                'id': str(uuid.uuid4()),
                'author': {
                    'role': 'user',
                },
                'content': content,
                'metadata': metadata,
            },
        ],
        'parent_message_id': str(uuid.uuid4()),
        'model': 'gpt-4',
        'plugin_ids': [],
        'timezone_offset_min': -480,
        'suggestions': [],
        'history_and_training_disabled': False,
        'arkose_token': None,
        'force_paragen': False,
    }

    headers = {
        'accept': '*/*',
        'accept-language': 'en-US',
        'authorization': f'Bearer {password}',
        'content-type': 'application/json',
        'dnt': '1',
        'origin': 'https://mirror.llm.beauty',
        'referer': 'https://mirror.llm.beauty/?model=gpt-4',
        'sec-ch-ua': '"Not=A?Brand";v="99", "Chromium";v="118"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
    }

    response = requests.post(
        'https://mirror.llm.beauty/backend-api/conversation',
        headers=headers,
        json=json_data,
        # verify=False,
    )

    return response.text

def get_conversation(password, conversation_id):
    headers = {
        'Accept-Language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        'Authorization': f'Bearer {password}',
        'Connection': 'keep-alive',
        'Content-Type': 'application/json',
        'DNT': '1',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        'accept': 'text/event-stream',
    }

    json_data = {
        'about_user_message': '',
        'about_model_message': '',
        'enabled': False,
    }

    response = requests.get(
    f'https://mirror.llm.beauty/backend-api/conversation/{conversation_id}',
        headers=headers,
        verify=False,
    )

    return response.text

def ask(prompt, image_path, access_token):
    response = None
    # access_token = r.lpop('access_token').decode()
    # access_token = GPT4VTOKENS.pop(0)
    try:
        file_data = open(image_path, 'rb').read()
        width, height = Image.open(image_path).size

        file_id = put_file(access_token, 'misc.jpg', file_data)

        # print(file_id, flush=True)
        stream_data = ask_code(access_token, content={
                            'content_type': 'multimodal_text',
                            'parts': [
                                {
                                    'asset_pointer': f'file-service://{file_id}',
                                    'size_bytes': len(file_data),
                                    'width': width,
                                    'height': height,
                                },
                                prompt,
                            ],
                        },metadata={
                'attachments': [
                    {
                        'name': 'image.png',
                        'id': file_id,
                        'size': len(file_data),
                        'mimeType': 'image/png',
                        'width': width,
                        'height': height,
                    },
                ],
            },)
        finall_data_raw = None
        for line in stream_data.split("\n"):
            if line.strip() == "data: [DONE]":
                break
            if line.startswith("data: "):
                line = line[len("data: "):]
                if "finished_successfully" in line:
                    finall_data_raw = line
        if finall_data_raw:
            finall_data = json.loads(finall_data_raw)
            if finall_data['message']['author']['role'] == 'assistant':
                response = finall_data['message']['content']['parts'][0]
        else:
            print(stream_data)
    except:
        traceback.print_exc()
    # r.rpush('access_token', access_token)
    # GPT4VTOKENS.append(access_token)
    return response



class GPT4VInterface():
    def __init__(self) -> None:
        self.TOKENS = GPT4VTOKENS
    
    def get_response(self, prompt, image_path):
        run_out = len(self.TOKENS)
        while True and run_out>0:
            run_out -= 1
            access_token = self.TOKENS.pop(0)
            try:
                response = ask(prompt, image_path, access_token)
                self.TOKENS.append(access_token)
                return 200, response
            except Exception as e:
                print(e)

        return -1, None





if __name__ == '__main__':
    from data.utils.template_util import *
    default_prompt = get_prompt(question="How many pictures are stored in this camera?", shot=4)

    default_prompt = """Given a image and an absurd question about the given image (the question usually asks about non-existent objects in the picture), please generate a multi-step reasoning chain to refute the question. Please output the generation result as a json with the format of{"steps": [xxx, xxx, ...], "conclusion": xxx}.

Q: What is the jersey number of the athlete raising both hands in front of the goal?
     """
    print(default_prompt)

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default=default_prompt)
    parser.add_argument('--image_path', type=str, default="./1245.jpg")
    args = parser.parse_args()

    result = ask(args.prompt, args.image_path, GPT4VTOKENS[0])
    print(result)


