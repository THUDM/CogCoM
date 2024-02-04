## Automatic Data Production Pipe-line for Chain of Manipulations (CoM)

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