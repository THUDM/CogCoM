"""
This script is a simple web demo of the CogCoM models, designed for easy and quick demonstrations.

Usage:
- Use the interface to upload images and enter text prompts to interact with the models.

Requirements:
- Gradio (only 3.x,4.x is not support) and other necessary Python dependencies must be installed.
- Proper model checkpoints should be accessible as specified in the script.

Note: This demo is ideal for a quick showcase of the CogVLM and CogAgent models. For a more comprehensive and interactive
experience, refer to the 'composite_demo'.
"""
import gradio as gr
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import torch
import time
from sat.mpu import get_model_parallel_world_size

from models.cogcom_model import CogCoMModel
from utils import chat
from utils import get_image_processor, llama2_tokenizer, llama2_text_processor_inference, parse_response




DESCRIPTION = '''<h1 style='text-align: center'> <a href="https://github.com/THUDM/CogCoM">CogCoM</a> </h1>'''

NOTES = '<h3> This app is adapted from <a href="https://github.com/THUDM/CogCoM">https://github.com/THUDM/CogCoM</a>. It would be recommended to check out the repo if you want to see the detail of our model, CogCoM. </h3>'

MAINTENANCE_NOTICE1 = 'Hint 1: If the app report "Something went wrong, connection error out", please turn off your proxy and retry.<br>Hint 2: If you upload a large size of image like 10MB, it may take some time to upload and process. Please be patient and wait.'


COM_NOTICE = 'Hint 1: To explicitly perform <strong> Grounding, Captioning, OCR, CoM</strong>, please use the <a href="https://github.com/THUDM/CogCoM#Cookbook">prompts for details</a>.'
# GROUNDING_NOTICE = 'Hint 1: To use <strong>Explicitly Launching CoM</strong>, please use the <a href="https://github.com/THUDM/CogCoM/blob/main/utils/com_dataset.py#L17">prompts for details</a>.'




default_chatbox = [("", "Hi, What do you want to know about this image?")]


model = image_processor = text_processor_infer = None

is_grounding = False

def process_image_without_resize(image_prompt):
    image = Image.open(image_prompt)
    # print(f"height:{image.height}, width:{image.width}")
    timestamp = int(time.time())
    file_ext = os.path.splitext(image_prompt)[1]
    filename_grounding = f"examples/{timestamp}_grounding{file_ext}"
    return image, filename_grounding

from sat.quantization.kernels import quantize

def load_model(args): 
    # load model
    model, model_args = CogCoMModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
        deepspeed=None,
        local_rank=rank,
        rank=rank,
        world_size=world_size,
        model_parallel_size=world_size,
        mode='inference',
        skip_init=True,
        use_gpu_initialization=True if (torch.cuda.is_available() and args.quant is None) else False,
        device='cuda',
        **vars(args)
    ), url='local', overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {})
    model = model.eval()
    from sat.mpu import get_model_parallel_world_size
    assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type="chat")
    image_processor = get_image_processor(490)
    cross_image_processor = get_image_processor(model_args.cross_image_pix) if "cross_image_pix" in model_args else None
    text_processor_infer = llama2_text_processor_inference(tokenizer, args.max_length, model.image_length if hasattr(model, 'image_length') else 0, model, False, english=True)


    return model, image_processor, cross_image_processor, text_processor_infer


def post(
        input_text,
        temperature,
        top_p,
        top_k,
        image_prompt,
        result_previous,
        hidden_image,
        state
        ):
    result_text = [(ele[0], ele[1]) for ele in result_previous]
    for i in range(len(result_text)-1, -1, -1):
        if result_text[i][0] == "" or result_text[i][0] == None:
            del result_text[i]
    print(f"history {result_text}")
    
    global model, image_processor, cross_image_processor, text_processor_infer, is_grounding

    try:
        with torch.no_grad():
            pil_img, image_path_grounding = process_image_without_resize(image_prompt)
            response, history, ret_imgs = chat(
                image_path="", 
                model=model, 
                text_processor=text_processor_infer,
                img_processor=image_processor,
                query=input_text, 
                history=result_text, 
                cross_img_processor=cross_image_processor,
                image=pil_img, 
                max_length=2048, 
                top_p=top_p, 
                temperature=temperature,
                top_k=top_k,
                invalid_slices=text_processor_infer.invalid_slices if hasattr(text_processor_infer, "invalid_slices") else [],
                parse_result=True
            )
    except Exception as e:
        print("error message", e)
        result_text.append((input_text, 'Timeout! Please wait a few minutes and retry.'))
        return "", result_text, hidden_image

    answer = response
    drawn_imgs = []
    # if is_grounding:
        # parse_response(pil_img, answer, image_path_grounding)
        # new_answer = answer.replace(input_text, "")
        # result_text.append((input_text, answer))
        # result_text.append((None, (image_path_grounding,)))
    # drawn_imgs = [(im[-1], f'trun-{i}') for i,im in enumerate(ret_imgs) if im[-1] is not None]
    drawn_imgs = [ret_imgs[-1]] if ret_imgs[-1] is not None else []

    # else:
    result_text.append((input_text, answer))
    print(result_text)
    print('finished')
    return "", result_text, hidden_image, drawn_imgs


def clear_fn(value):
    return "", default_chatbox, None

def clear_fn2(value):
    return default_chatbox


def main(args):
    global model, image_processor, cross_image_processor, text_processor_infer, is_grounding
    model, image_processor, cross_image_processor, text_processor_infer = load_model(args)
    # is_grounding = 'grounding' in args.from_pretrained
    
    gr.close_all()

    with gr.Blocks(css='style.css') as demo:
        state = gr.State({'args': args})

        gr.Markdown(DESCRIPTION)
        gr.Markdown(NOTES)
        

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Group():
                    gr.Markdown(COM_NOTICE)
                    # gr.Markdown(GROUNDING_NOTICE)
                    input_text = gr.Textbox(label='Input Text', placeholder='Please enter text prompt below and press ENTER.')
                    
                    with gr.Row():
                        run_button = gr.Button('Generate')
                        clear_button = gr.Button('Clear')

                    image_prompt = gr.Image(type="filepath", label="Image Prompt", value=None)

                with gr.Row():
                    temperature = gr.Slider(maximum=1, value=0.8, minimum=0, label='Temperature')
                    top_p = gr.Slider(maximum=1, value=0.4, minimum=0, label='Top P')
                    top_k = gr.Slider(maximum=100, value=10, minimum=1, step=1, label='Top K')

            with gr.Column(scale=4):
                result_text = gr.components.Chatbot(label='Multi-round conversation History', value=[("", "Hi, What do you want to know about this image?")], height=600)
                hidden_image_hash = gr.Textbox(visible=False)
            
            with gr.Column(scale=2):
                drawn_imgs = gr.Gallery(
                    label="Resultant images", show_label=False, elem_id="gallery", columns=[1], rows=[4], object_fit="contain", width="auto")


        gr.Markdown(MAINTENANCE_NOTICE1)

        print(gr.__version__)
        run_button.click(fn=post,inputs=[input_text, temperature, top_p, top_k, image_prompt, result_text, hidden_image_hash, state],
                         outputs=[input_text, result_text, hidden_image_hash, drawn_imgs])
        input_text.submit(fn=post,inputs=[input_text, temperature, top_p, top_k, image_prompt, result_text, hidden_image_hash, state],
                         outputs=[input_text, result_text, hidden_image_hash, drawn_imgs])
        clear_button.click(fn=clear_fn, inputs=clear_button, outputs=[input_text, result_text, image_prompt])
        image_prompt.upload(fn=clear_fn2, inputs=clear_button, outputs=[result_text])
        image_prompt.clear(fn=clear_fn2, inputs=clear_button, outputs=[result_text])


    # demo.queue(concurrency_count=10)
    demo.launch(server_port=7190)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    parser.add_argument("--version", type=str, default="chat", choices=['chat', 'vqa', 'chat_old', 'base'], help='version of language process. if there is \"text_processor_version\" in model_config.json, this option will be overwritten')
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None, help='quantization bits')
    parser.add_argument("--from_pretrained", type=str, default="cogcom-base", help='pretrained ckpt')
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--stream_chat", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    args = parser.parse_args()   
    main(args)