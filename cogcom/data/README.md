## Run the Automatic Data Generation Pipeline for Chain of Manipulations (CoM)

<p align="center">
    <img src="https://raw.githubusercontent.com/THUDM/CogCoM/master/assets/data_framework.png" width=100% style="margin-bottom: 0.2;"/>
<p>

### 1. Prepare raw data  
This step prepare the original data source to a unified data format.

 - Put (or link) all your data into the `save/raw` directory according to the scripts in `prepare` folder
 - Just run the following command

 ```bash
    `bash run_prepare.sh
 ```

This command will save the processed results in to `save/processed`.



### 2. Generate linguistic solving steps
This step generate the linguistic solving steps based on GPT4. In addition, we can also generate solving steps for absurd visual question to resist hallucinations based on GPT4-V.

 - You need to configure the API tokens for GPT4 and GPT4-V in the scripts that we have written to call these LLMs, at `tools/gpt4.py` and `tools/gpt4v.py`
 - Run the following python scripts for generation, this might spend **several days** accroding to the parallel processes.

 ```python
    python gen_steps_txt.py
    python pro_steps_txt.py
 ```

 ```python
    python gen_absurd.py
    python pro_steps_absurd.py
 ```

These command will save the generated results into the directors suffixed with `_extract`.


### 3. Compensate visual annotations
This step compensate the visual contents requested by the manipulations in previous generate solving steps.

 - Just run the following command

 ```bash
    bash run_visual_ann.sh
 ```

This command will save the compensated results into the directories suffixed with `_visual`.


### 4. Convert data to WebDataset format
As we train CogCoM using the WebDataset data processor, we can easily convert the generated results to this format.

 - Just run the following command.

 ```bash
    bash run_convert_wds.sh
 ```

This command will convert all generated results in previous step to the WebDataaset format that we can using it to train CogCoM. The save the directories will be suffixed with `_wds`.



## Download our Prepared Data

We have also open-sourced the data we built to train CogCoM to facilitate potential research, which includes:
  - Instruction-tuning datasets (MultiInstruct-366K, ShareGPT4V-100K, LLaVAR-34K, ShikraGrounding-14K)
  - Automatically synthesized CoM data (84K positive chains)
  - Automatically synthesized CoM-test data (8K positive chains)
  - Manually annotated CoM-Math data  (7K positive chains)


### Data Downloading

  - Download the **Instruction-tuning datasets** prepared with the `WebDataset` format (including serialized image bytes) [here](https://cloud.tsinghua.edu.cn/d/07e5a8d3bdcd4bb18ff3/).
  - Download all **CoM datasets**, including both automatically synthesized data and manually annotated math data from [here](https://huggingface.co/datasets/qijimrc/CoMDataset).


### Data Examples

Examples of (1) our automatically synthesized data and (2) our manually annotated math data. 

<img src="https://raw.githubusercontent.com/THUDM/CogCoM/master/assets/eg_comdata.png" style="zoom:100%;" />




Each data sample in the dataset is provided in json format and contains the following attributes:

```json
{
    "pid": "[int] Problem ID, e.g., 1",
    "image_path": "[string] A file path pointing to the associated image",
    "question": "[string] The question text",
    "answer": "[string] The correct answer for the problem",
    "com_founds": "[list] the tree nodes where the golden answer was found",
    "final_com": {
        "a,b--c,d": // a: parent's level, b: parent's index, c: current node's level, current node's index,
        {
            "func": "[string] the current manipulation function",
            "param": "[string] the input parameter of current manipulation",
            "onbox": "[list] bounding boxes where current manipulation will operate on",
            "variables": "[dict] mappings from placeholders to real values in `desc`",
            "desc": "[string] the textual description of current reasoning step",
            "return": "[list] the return value of current manipulation",
            "found": "[bool] whether the golden answer is found at current node",
        },
    },
    "cropped": "[bool]  whether the CropZoomIn manipulation is used",
}
```

### Data Visualization

You can view the CoM samples with reasoning chains using our visualization script `/cogcom/data/utils/visualize.ipynb`

<details>
<summary>Click to expand/collapse the visualization page screeshot.</summary>
<img src="https://raw.githubusercontent.com/THUDM/CogCoM/master/assets/eg_comtest.png" style="zoom:100%;" />
<img src="https://raw.githubusercontent.com/THUDM/CogCoM/master/assets/eg_commath326.png" style="zoom:100%;" />
<img src="https://raw.githubusercontent.com/THUDM/CogCoM/master/assets/eg_commath20.png" style="zoom:100%;" />
</details>

### Data Source

The **CoM** and **CoM-test** datasets are derived from existing public datasets: ST-VQA, TextVQA, and TDIUC. The **CoM-Math** dataset is derived and further manually annotated from the MathVista dataset. Details can be found in the [paper](https://arxiv.org/pdf/2402.04236). All these source datasets have been preprocessed and labeled for training and evaluation purposes.


### License

The new contributions to our dataset are distributed under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license, including

- The creation of three datasets: CoM, CoM-test, and CoM-Math;
- The filtering and cleaning of source datasets;
- The standard formalization of instances for evaluation purposes;
- The annotations of metadata.

The copyright of the images, questions and the answers belongs to the original authors. Alongside this license, the following conditions apply:

- **Purpose:** The dataset was primarily designed for use as training sets and test sets.
- **Commercial Use:** The dataset can be used commercially as training sets and test sets. By accessing or using this dataset, you acknowledge and agree to abide by these terms in conjunction with the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.
