# This repo contains all th enecessary files for the Zindi Competition

**Lelapa AI Zindi Notebook - Participants.ipynb** 
This notebook is one that can be shared with participants.  
[Lelapa AI Zindi Notebook - Participants.ipynb](https://github.com/Lelapa-AI/zindi-inkuba-notebook/blob/main/Lelapa_AI_Zindi_Notebook_Participants.ipynb)

**Lelapa AI Zindi Score - Zindi team **

This notebook if for Zindi Admin. It includes a code snippet at the end to combine submission file with groundtruth to compute Zindi score, It does also have the rest of the partcipant code for generating submission files, note that these are not necessary to do the scoring
[Lelapa AI Zindi Score - Zindi team.ipynb](https://github.com/Lelapa-AI/zindi-inkuba-notebook/blob/main/Lelapa_AI_Zindi_Score_Zindi_team.ipynb) 

**Helper Functions that both Zindi team and Participants need access to**

[eval.py](https://github.com/Lelapa-AI/zindi-inkuba-notebook/blob/main/eval.py) -> this file contains all the evaluation functions
[model_function.py](https://github.com/Lelapa-AI/zindi-inkuba-notebook/blob/main/model_function.py) -> this file contains functions for model inference

**Helper files that only Zindi Admin would utilise**

[zindi_scorer.py](https://github.com/Lelapa-AI/zindi-inkuba-notebook/blob/main/zindi_scorer.py) -> contains submission evaluation functions

### Example Submission files:

This files are made from running the participant notebook on different pre-existing models on huggingface

* InkubaLM (lelapa/InkubaLM-0.4B) [Submission file](https://github.com/Lelapa-AI/zindi-inkuba-notebook/tree/main/submission_1)
* SmoLlm (HuggingFaceTB/SmolLM-135M) Submisison file: [Submisison files](https://github.com/Lelapa-AI/zindi-inkuba-notebook/tree/main/submission_2)
* TinyLlama (TinyLlama/TinyLlama_v1.1) Submisison file: 
* Qwen (Qwen/Qwen2-1.5B) Submisison file: 

### All the other csv files
These are the interim files used in the submission file creation process so you can see how evaluations might work for eacj langauge and task. These files would be generated on the participant side when they are evalauting the model (we have included them here just for funsies so you do not have to run everything to test)
