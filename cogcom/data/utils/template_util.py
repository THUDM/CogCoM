import numpy as np
from pathlib import Path
import json
import os


# with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates_v1.json')) as f:
with open(Path(__file__).parent/'templates_v2.json') as f:
    TEMPLATE = json.load(f)


def get_manipulations_prompt():
    """
      (1) manipulation: description.
      (2) manipulatoon: description.
    """
    all_mps = []
    for i, mp in enumerate(TEMPLATE['Manipulations']['on_1_image']):
        mp_txt = "({}) {}: {}".format(i+1, mp['name'], mp['description'])
        all_mps.append(mp_txt)
    prompt = ";\n".join(all_mps)
    return prompt

def get_prompt(question, shot=0):
    manipulation_prompt = get_manipulations_prompt()

    if shot == 0:
        prompt = TEMPLATE['Prompts']['0shot'].format(
            MANIPULATIONS = manipulation_prompt,
            QUESTION = question
        )
    elif shot == 1:
        # demo = TEMPLATE['Demonstrations'][-1]
        demo = TEMPLATE['Demonstrations'][0]
        q, steps = demo['question'], demo['steps']
        demo_prompt = "Q: {question} The solving steps are: {step}".format(
                question = q,
                step = "; ".join([f"Step {i+1}: ({stp['manipulation']}, {stp['text']})" for i, stp in enumerate(steps)])
            )
        prompt = TEMPLATE['Prompts']['1shot'].format(
            MANIPULATIONS = manipulation_prompt,
            DEMONSTRATIONS = demo_prompt,
            QUESTION = question
        )
    else:
        n = min(shot, len(TEMPLATE['Demonstrations']))
        demo_prompt = ""
        for demo in np.random.choice(TEMPLATE['Demonstrations'], n, replace=False):
            q, steps = demo['question'], demo['steps']
            prompt = ""
            demo_prompt += "Q: {question} The solving steps are: {step}\n".format(
                question = q,
                step = "; ".join([f"Step {i+1}: ({stp['manipulation']}, {stp['text']})" for i, stp in enumerate(steps)])
            )
        prompt = TEMPLATE['Prompts']['nshot'].format(
            MANIPULATIONS = manipulation_prompt,
            DEMONSTRATIONS = demo_prompt,
            QUESTION = question
        )
    return prompt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', type=str, default="How many pople are in this office room?")
    parser.add_argument('--shot', type=int, default=7)
    args = parser.parse_args()

    reponse = get_prompt(args.question, shot=args.shot)
    print(reponse)