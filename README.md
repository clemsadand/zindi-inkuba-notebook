# This repo contains all the necessary files for the Zindi Competition

**1. Lelapa AI Zindi Notebook - Participants.ipynb** 
This notebook is one that can be shared with participants.  
[Lelapa AI Zindi Notebook - Participants.ipynb](https://github.com/Lelapa-AI/zindi-inkuba-notebook/blob/main/Lelapa_AI_Zindi_Notebook_Participants.ipynb)

- it takes the partcipants through how to access the data and run inference on the dev set
- it shows how their dev set can be scored (so they can chec their progress)
- it has the code to create the two column subission file

**2. Lelapa AI Zindi Score - Zindi team.ipynb** [Lelapa AI Zindi Score - Zindi team.ipynb](https://github.com/Lelapa-AI/zindi-inkuba-notebook/blob/main/Lelapa_AI_Zindi_Score_Zindi_team.ipynb)

This notebook if for Zindi Admin. It includes a code snippet at the end to combine submission file with groundtruth to compute Zindi score, It does also have the rest of the partcipant code for generating submission files, note that these are not necessary to do the scoring
- this notebook shows how to create the target files for the public scoreboard and the private scoreboard (though these files have been included in the repo so those can be used instead)
- this notebook also shows how to combine the target with the submission file to create files to score for the public and private scoreoard (though we imagine Zindi has their own process for this so this is just an exmaple)
- it then shows how the two cilumn submission files when paired with the target can be scored
- the Zindi scorer file has a few imports (List as a data type import which is not necessary), it also has the COunter import and the CSV import. We have otherwise coded the Chrf and F1score from scratch (hope this helps for future Zindi challenges too)

**3. Helper Functions that Participants need access to**

[eval.py](https://github.com/Lelapa-AI/zindi-inkuba-notebook/blob/main/eval.py) -> this file contains all the evaluation functions
[model_function.py](https://github.com/Lelapa-AI/zindi-inkuba-notebook/blob/main/model_function.py) -> this file contains functions for model inference necessary for the submission format

**4. Helper files that only Zindi Admin would utilise**

[zindi_scorer.py](https://github.com/Lelapa-AI/zindi-inkuba-notebook/blob/main/zindi_scorer.py) -> contains submission evaluation functions

**5. Example Submission files:**

This file is made from running the participant notebook on InkubaLM

### All the other csv files
These are the interim files used in the submission file creation process so you can see how evaluations might work for each langauge and task. These files would be generated on the participant side when they are evalauting the model (we have included them here just for funsies so you do not have to run everything to test). There are two working files created on the scoring side after merging the target dataset and submission file togther as well. These are the example files of how the scorer function works for thee public scoreboard and the private scoreboard. 

# Zindi Score/Metric
The main point of the challenge is to make the model smaller and or smarter. The final score combines the size of the model and the model performance. The zindi score takes the avergae performance of the model and you have the chance to double this score by by making the model 100% smaller than Inkuba. 

### Size comparison
We utilise the number of weights of the model to compare the size. At the end of the day it is the number of weights that impact inference time of the model and the fewer the number of weights, the more accessible the model is in low reosurce environments. 

### Model Performance Metric
The model perfromance is a combination of multiple metrics because the model is a general base model capable of perfroming multiple tasks in multiple languages. We have chosen Yoruba and Swhaili as the two languages and the tasks are Sentiment Analysis, AfriXNLI (true, false, neiher) as well as Machine Translation in the Eng-> African langugae direction. Sentiment and AfriXnli accuracy/f1 are calculated using logliklihood. Machine translation is calculated using the CHRF metric. All metric are out of 100. The averga model performance score is the avergae accross these metrics.

### Combined metric for size and model performance
The final score is a score out of 1 (which can be converted to a percentage out of 100). The score is a combination of the model perfromance scaled by the size ration of the submitted model with the original Inkuba model. Essentially, a users model perfrmance score will be doubled if the model submitted is twice as small as Inkuba, the score would not change if the model is the same size as Inkuba and the score will be negative if the model is bigger than Inkuba. We are only interested in models that are the same size or smaller. 

