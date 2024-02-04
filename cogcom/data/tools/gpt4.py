import os, sys
sys.path.append(os.path.dirname(__file__))
import argparse
import openai
from utils.template_util import *


openai.api_key = ''
openai.api_base = ""


class GPT4PI:
    def __init__(self) -> None:
        pass


    def get_response(self, prompt):
        # create a chat completion
        try:
            rt_completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )

            status, response = -1, ""
            if len(rt_completion.choices) > 0:
                response = rt_completion.choices[0].message.content
                status = 200
        except Exception as e:
            print(e)
            import ipdb
            ipdb.set_trace()
            status, response = -1, ""
        return status, response

if __name__ == '__main__':

    default_prompt = get_prompt(question="How old is the man standing behind the black table?", shot=7)
    print(default_prompt)

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default=default_prompt)
    args = parser.parse_args()

    chat_api = GPT4PI()
    resp = chat_api.get_response(None, prompt=args.prompt)
    print(resp)
