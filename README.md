# CogCoM

<!-- 📗 [中文版README](./README_zh.md) -->
🆕 ```2024/2/4```: Release the base model CogCoM-base-17b.

🌟 Jump to detailed introduction: [Introduction to CogCoM](#introduction-to-cogcom).


<table>
  <tr>
    <td>
      <h2> CogCoM </h2>
      <p> 📖  Paper: <a href="https://arxiv.org/">CogCoM: Train Large Vision-Language Models Diving into Details through Chain of Manipulations</a></p>
      <p><b>CogCoM</b> is a general vision-language model (VLM) endowed with Chain of Manipulations (CoM) mechanism, that enables VLMs to perform multi-turns evidential visual reasoning by actively manipulating the input image. We now release CogCoM-base-17b, a model with 10 billion visual parameters and 7 billion language parameters, trained on a data fusion of 4 types capabilities (instruction-following, OCR, detailed-captioning, and CoM).</p>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <p>🌐 Web Demo is coming soon.</a></p>
    </td>
  </tr>
</table>


**Table of Contents**

- [CogCoM](#cogcom)
    - [Release](#release)
    - [Get Started](#get-started)
        - [Option 1: Inference Using Web Demo.](#option-1-inference-using-web-demo)
        - [Option 2：Deploy CogCoM by yourself](#option-2-deploy-cogcom-by-yourself)
            - [Situation 2.1 CLI (SAT version)](#situation-21-cli-sat-version)
            - [Situation 2.2 CLI (Huggingface version)](#situation-22-cli-huggingface-version)
            - [Situation 2.3 Web Demo](#situation-23-web-demo)
        - [Option 3：Finetuning CogCoM](#option-3finetuning-cogacom)
        - [Hardware requirement](#hardware-requirement)
        - [Model checkpoints](#model-checkpoints)
    - [Introduction to CogCoM](#introduction-to-cogcom)
        - [Examples](#examples)
    - [Cookbook](#cookbook)
        - [Task Prompts](#task-prompts)
        - [Which --version to use](#which---version-to-use)
        - [FAQ](#faq)
    - [License](#license)
    - [Citation \& Acknowledgements](#citation--acknowledgements)

## Release
- ```2024/2/4``` CogCoM-base-17b released.

## Get Started

### Option 1: Inference Using Web Demo.

* Now you can use the local code we have implemented with Gradio for [GUI demo](/cogcom/demo/web_demo.py). The web demo is coming soon.


### Option 2：Deploy CogCoM by yourself

We support two GUIs for model inference, **CLI** and **web demo** . If you want to use it in your python code, it is
easy to modify the CLI scripts for your case.


First, we need to install the dependencies.

```bash
# CUDA >= 11.8
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**All code for inference is located under the ``demo/`` directory. Please switch to this directory first before proceeding with further operations.**

#### Situation 2.1 CLI (SAT version)

Run CLI demo via:

```bash
python cli_demo_sat.py --from_pretrained cogcom-base-17b --local_tokenizer path/to/tokenizer --bf16 --english
```

The program will automatically download the sat model and interact in the command line (can simply using vicuna-7b-1.5 tokenizer). You can generate replies by
entering instructions and pressing enter.
Enter `clear` to clear the conversation history and `stop` to stop the program.

We also support model parallel inference, which splits model to multiple (2/4/8) GPUs. `--nproc-per-node=[n]` in the
following command controls the number of used GPUs.

```
torchrun --standalone --nnodes=1 --nproc-per-node=2 cli_demo_sat.py --from_pretrained cogcom-base-17b --local_tokenizer path/to/tokenizer --bf16
```

- If you want to manually download the weights, you can replace the path after ``--from_pretrained`` with the model
  path.

- Our model supports SAT's **4-bit quantization** and **8-bit quantization**.
  You can change ``--bf16`` to ``--fp16``, or ``--fp16 --quant 4``, or ``--fp16 --quant 8``.

  For example

    ```bash
    python cli_demo_sat.py --from_pretrained cogcom-base-17b --fp16 --quant 8
    # In SAT version，--quant should be used with --fp16
    ```

- The program provides the following hyperparameters to control the generation process:
    ```
    usage: cli_demo_sat.py [-h] [--max_length MAX_LENGTH] [--top_p TOP_P] [--top_k TOP_K] [--temperature TEMPERATURE]

    optional arguments:
        -h, --help                    show this help message and exit
        --max_length MAX_LENGTH       max length of the total sequence
        --top_p TOP_P                 top p for nucleus sampling
        --top_k TOP_K                 top k for top k sampling
        --temperature TEMPERATURE     temperature for sampling
    ```

#### Situation 2.2 CLI (Huggingface version)

Run CLI demo via:

```bash
# CogCoM
python cli_demo_hf.py --from_pretrained THUDM/cogcom-base-17b-hf --bf16 --local_tokenizer path/to/tokenizer --bf16 --english
```

- If you want to manually download the weights, you can replace the path after ``--from_pretrained`` with the model
  path.

- You can change ``--bf16`` to ``--fp16``, or ``--quant 4``. For example, our model supports Huggingface's **4-bit
  quantization**:

    ```bash
    python cli_demo_hf.py --from_pretrained THUDM/cogcom-base-17b-hf --quant 4
    ```

#### Situation 2.3 Web Demo

We also offer a local web demo based on Gradio. First, install Gradio by running: `pip install gradio`. Then download
and enter this repository and run `web_demo.py`. See the next section for detailed usage:

```bash
python web_demo.py --from_pretrained cogcom-base-17b --local_tokenizer path/to/tokenizer --bf16 --english
```

The GUI of the web demo looks like:

<div align="center">
    <img src=assets/web_demo-min.png width=70% />
</div>

### Option 3：Finetuning CogCoM

You may want to use CogCoM in your own task, which needs a **different output style or domain knowledge**. **All code
for finetuning is located under at ``finetune.sh`` and ``finetune.py`` files.**


### Hardware requirement

* Model Inference:

  For INT4 quantization: 1 * RTX 3090(24G)

  For FP16: 1 * A100(80G) or 2 * RTX 3090(24G)

* Finetuning:

  For FP16: 4 * A100(80G) *[Recommend]* or 8* RTX 3090(24G).

### Model checkpoints

If you run the `demo/cli_demo*.py` from the code repository, it will automatically download SAT or Hugging Face
weights. Alternatively, you can choose to manually download the necessary weights.


- CogCoM

  |          Model name           | Input resolution |                           Introduction                            | Huggingface model | SAT model |
  | :-------------------------: | :----: | :-------------------------------------------------------: | :------: | :-------: |
  |         cogcom-base-17b         |  490   |  Supports chat, grounding, OCR, and CoM.   |  [link](https://huggingface.co/THUDM)        |    [link](https://huggingface.co/THUDM)        |

## Introduction to CogCoM

- CogCoM is a general **open-source visual language model** (**VLM**) equipped with Chain of Manipulations (CoM). CogCoM-17B has 10 billion vision parameters and
  7 billion language parameters.
- CogCoM-17B rely on an efficient CoM data production framework, that engages remarkable LLM to provide basic solving steps, adopts reliable visual tools to obtain visual contents, and then acquires feasible paths based on traversal.
- CogCoM-17B is trained on a data fusion of 4 types capabilities, including instruction-following, OCR, detailed-captioning, and CoM, which can solve general multimodal tasks and can perform evidential visual reasoning that permits uses to trace the error causes in the interpretable paths.
- CogCoM devises a memory-based compatible VLM architecture, that enables VLMs to actively manipulate the input image (e.g., grounding, crop, zoom in) and re-input the processed new iamge with a multi-turns multi-images manner, for rigorously reasoning.


<div align="center">
    <img src=assets/framework.jpg width=70% />
</div>



<details>
<summary>Click to view results on GQA, TallyVQA, TextVQA, ST-VQA. </summary>

<table>
    <tr>
        <td>Method</td>
        <td>GQA</td>
        <td>TallyVQA-s</td>
        <td>TallyVQA-c</td>
        <td>TextVQA</td>
        <td>ST-VQA</td>
    </tr>
    <tr>
        <td>Flamingo</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>54.1</td>
        <td>-</td>
    </tr>
     <tr>
        <td>GIT</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>59.8</td>
        <td>-</td>
    </tr>
     <tr>
        <td>GIT2</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>67.3</td>
        <td>-</td>
    </tr>
    <tr>
        <td>BLIP-2</td>
        <td>44.7*</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>21.7</td>
    </tr>
    <tr>
        <td>InstructBLIP</td>
        <td>49.5*</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>50.7*</td>
    </tr>
    <tr>
        <td>Qwen-VL</td>
        <td>49.5*</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>50.7*</td>
    </tr>
    <tr>
        <td>CogCoM</td>
        <td>71.7</td>
        <td>84.0</td>
        <td>70.1</td>
        <td>71.1</td>
        <td>70.0</td>
    </tr>
</table>

</details>

<details>
<summary>Click to view results of grounding benchmarks. </summary>

<table>
    <tr>
        <td></td>
        <td>RefCOCO</td>
        <td></td>
        <td></td>
        <td>RefCOCO+</td>
        <td></td>
        <td></td>
        <td>RefCOCOg</td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td>val</td>
        <td>testA</td>
        <td>testB</td>
        <td>val</td>
        <td>testA</td>
        <td>testB</td>
        <td>val</td>
        <td>test</td>
    </tr>
    <tr>
        <td>CogCoM-grounding-generalist</td>
        <td>92.34</td>
        <td>94.57</td>
        <td>89.15</td>
        <td>88.19</td>
        <td>92.80</td>
        <td>82.08</td>
        <td>89.32</td>
        <td>90.45</td>
    </tr>
</table>
</details>

### Examples


* CogCoM performs evidential visual reasoning for details recognition, reading time, understanding charts, counting objects, and reading texts.
    <details open>
    <summary>Click for view examples.</summary>
    <div align="center">
    <img src=assets/cases.jpg width=70% />
    </div>

    </details>
    <br>
    <br>

* CogCoM demonstrates the flexible capabilities for adapting to different multimodal scenarios, including evidential visual
reasoning, Visual Grounding, Grounded Captioning, Image Captioning, Multi Choice, and Detailed Captioning.

<div align="center">
    <img src=assets/app_case.jpg width=70% />
</div>





## Cookbook

### Task Prompts

1. **General Multi-Round Dialogue**: Say whatever you want.

2. **Chain of Manipulations** : Explicitly launching CoM reasoning.

    - We randomly add launching prompts to the CoM chains for solving meticulous visual problems, so you can explicitly let CogCoM to run with CoM mechanism, by adding the following launching prompt (we randomly generate numerous launching prompts for flexibility, see `com_dataset.py` for all details):

    ```bash
        Please solve the problem gradually via a chain of manipulations, where in each step you can selectively adopt one of the following manipulations GROUNDING(a phrase)->boxes, OCR(an image or a region)->texts, CROP_AND_ZOOMIN(a region on given image)->new_image, CALCULATE(a computable target)->numbers, or invent a new manipulation, if that seems helpful. {QUESTION}
    ```


3. **Visual Grounding**. Our model is compatible with the grounding instructions from MultiInstruct and CogVLM, we provide basic usage of three functionalities here:

    - **Visual Grounding (VG)**: Returning grounding coordinates (bounding box) based on the description of objects. Use any template from [instruction template](cogcom/utils/template.py). For example (replacing <expr> with the object's description):

      > "Find the region in image that <expr> describes."

    - **Grounded Captioning (GC)**: Providing a description based on bounding box coordinates. Use a template from [instruction template](cogcom/utils/template.py). For example (replacing <objs> with the position coordinates),

      > "Describe the content of *[[086,540,400,760]]* in the picture."

    - **Image Description with Cooordinates (IDC)**: Image description with grounding coordinates (bounding box). Use any template
      from [caption_with_box template](https://github.com/THUDM/CogVLM/blob/main/utils/utils/template.py#L537) as model
      input. For example:

      > Can you provide a description of the image and include the coordinates [[x0,y0,x1,y1]] for each mentioned object?
    
**Format of coordination:** The bounding box coordinates in the model's input and output use the
format ``[[x1, y1, x2, y2]]``, with the origin at the top left corner, the x-axis to the right, and the y-axis
downward. (x1, y1) and (x2, y2) are the top-left and bottom-right corners, respectively, with values as relative
coordinates multiplied by 1000 (prefixed with zeros to three digits).


### FAQ

* If you have trouble in accessing huggingface.co, you can add `--local_tokenizer /path/to/vicuna-7b-v1.5` to load the
  tokenizer.
* Download model using 🔨[SAT](https://github.com/THUDM/SwissArmyTransformer), the model will be saved to the default
  location `~/.sat_models`. Change the default location by setting the environment variable `SAT_HOME`. For example, if
  you want to save the model to `/path/to/my/models`, you can run `export SAT_HOME=/path/to/my/models` before running
  the python command.

## License

The code in this repository is open source under the [Apache-2.0 license](./LICENSE), while the use of the CogCoM model
weights must comply with the [Model License](./MODEL_LICENSE).

