import pandas as pd
import datasets

"""
#you will need to install these necessary packeges
!pip install objsize
!pip install sacrebleu
!pip install --upgrade transformers accelerate sentencepiece datasets evaluate -q
"""

# Download data from huggingface that has the targets for the test set (no-one else will have access to the URL)
hau_sent  = datasets.load_dataset("lelapa/Zindi_sentiment_with_target", 'hausa')['test'].to_pandas()
swa_sent  = datasets.load_dataset("lelapa/Zindi_sentiment_with_target", 'swahili')['test'].to_pandas()
hau_mmt = datasets.load_dataset("lelapa/Zindi_eng_african_with_target", 'eng-hau')['test'].to_pandas()
swa_mmt = datasets.load_dataset("lelapa/Zindi_eng_african_with_target", 'eng-swa')['test'].to_pandas()
hau_xnli  = datasets.load_dataset("lelapa/Zindi_Afrixnli_with_target", "hau")['test'].to_pandas()
swa_xnli = datasets.load_dataset("lelapa/Zindi_Afrixnli_with_target", "swa")['test'].to_pandas()
res = pd.concat([hau_sent, swa_sent, hau_mmt, swa_mmt, hau_xnli, swa_xnli],ignore_index=True)

def get_zindi_score(submission_file):
    """
    This function pulls the test dataset from the huggingface server, merges the targets back onto the dataset and 
    runs the same evaluation function on a submission file to determine the Zindi score
    """
    data = pd.read_csv(submission_file)  #function takes in csv so this is csv 
    merged_submission = pd.concat([res, data], axis=1)
    merged_submission = merged_submission.rename(columns = {'targets':'Targets'}) 
    merged_submission.to_csv("submission_to_score.csv")
    zindi_score = eval.zindi_score("submission_to_score.csv")
    return (zindi_score)

score = get_zindi_score('submission_test.csv')
