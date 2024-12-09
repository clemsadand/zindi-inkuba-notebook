import csv
from collections import Counter
from statistics import mean
from typing import List

import evaluate
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PARAM_SIZE = 421939200.0


def zindi_score(submission_file):
    """
    This function generates the Lelapa AI Zindi score by combining model perfromance with model size
    This function relies on the structure of the file, so ensure no changes to the submission file structure
    are made

    Inputs:
    submission_file: <string> Path to submisison file csv

    Outputs: <float> Lelapa AI Zindi score that appears on the leader board
    """
    df = pd.read_csv(submission_file)
    avg_score = evaluate_zindi(df)
    size = df[df["Instruction"] == "Model size"]["Input Text"].astype(int)
    score = (avg_score + (1 - (size / PARAM_SIZE)) * avg_score) / 2  # wat
    return score


def calculate_chrf(df):
    """
    This function claculates CHRF when given a prediction and ground truth for translation

    Inputs:
    df: <pandas dataframe> with prediction and ground truth for machine translation

    Outputs:
    score: <dict> chrf results
    """
    chrf_metric = evaluate.load("chrf")
    references = format_references(df["Targets"].to_list())
    predictions = df["Response"].to_list()
    score = chrf_metric.compute(predictions=predictions, references=references)
    return score


def format_references(list_of_references: List) -> List:
    """
    This function formats the references in a List of List, as expected by the CHRF metric (and BLEU as well)
    """
    return [[reference] for reference in list_of_references]


def filter_predictions(prediction_df, task):
    # check if instruction is in prediction
    prediction_df["Response"] = (
        prediction_df["Response"]
        .str.split("<pad>")
        .str[-1]
        .str.split("</s>")
        .str[0]
        .str.strip()
    )
    prediction_df["Response"] = prediction_df.apply(
        remove_instruction_from_pred, axis=1
    )
    # normalise text
    prediction_df = prediction_df.fillna("unknown")
    prediction_df[["Targets", "Response"]] = prediction_df[
        ["Targets", "Response"]
    ].applymap(normalize_text)
    if task == "mt":
        return prediction_df

    # extract values based on task keyword (this needs to be updated)
    prediction_df["Response"] = prediction_df["Response"].apply(verbalizer)
    if task == "xnli":
        replacements = {"positive": "entailment", "negative": "contradiction"}
        prediction_df["Response"] = prediction_df["Response"].replace(replacements)
    prediction_df["Response"] = prediction_df.apply(assign_label, axis=1, task=task)

    return prediction_df


def process_likelihood(likelihood_str: str) -> List[float]:
    # clean the string to remove unwanted characters
    print(f"Raw Likelihood String: {likelihood_str}")
    clean_str = (
        likelihood_str.replace("tensor(", "").replace(")", "").strip()
    )  # remove 'tensor(' and ')'
    clean_str = (
        clean_str.replace("[[", "").replace("]]", "").strip()
    )  # remove extra brackets
    clean_str = (
        clean_str.replace(" device='cuda:0'", "")
        .replace(" dtype=torch.float16", "")
        .strip()
    )  # remove device and dtype info
    clean_str = clean_str.replace(
        "tensor", ""
    ).strip()  # remove any instances of 'tensor'

    # remove any empty strings caused by extra commas
    clean_str = clean_str.replace(",,", ",")  # remove duplicate commas if they exist

    # debugging output for verification
    print(f"Processed String: {clean_str}")  # debugging line

    # Convert to a list of floats
    likelihood = [
        float(x) for x in clean_str.split(",") if x.strip()
    ]  # ensure non-empty strings are converted
    return likelihood


def loglikelihoods_to_predictions(
    log_likelihoods: List[float], labels: List[str]
) -> List[str]:
    predictions = []

    for likelihoods in log_likelihoods:
        if isinstance(likelihoods, list):
            predicted_label_idx = np.argmax(likelihoods)
            predictions.append(labels[predicted_label_idx])
        else:
            print(f"Invalid likelihood format: {likelihoods}")
    return predictions


def compute_f1_and_accuracy(
    predictions: List[str], ground_truths: List[str], labels: List[str]
) -> (float, float):
    label_to_id = {label: i for i, label in enumerate(labels)}

    y_true = [label_to_id[label] for label in ground_truths]
    y_pred = [label_to_id[label] for label in predictions]

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    return accuracy, f1


def main(csv_file_path):
    log_likelihoods = []
    ground_truths = []

    with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        count = 0
        for row in reader:
            try:
                task = row["Task"]
            except:
                task = "xnli"
            lang = row["Langs"]
            if task == "sentiment" and lang == "swahili":
                labels = ["Chanya", "Wastani", "Hasi"]
            elif task == "sentiment" and lang == "hausa":
                labels = ["Kyakkyawa", "Tsaka-tsaki", "Korau"]
            else:
                labels = ["0", "1", "2"]
            # likelihood = [float(x) for x in row['Log-Likelihood'].strip('[]').split(',')]
            likelihood = process_likelihood(row["Log-Likelihood"])
            log_likelihoods.append(likelihood)
            ground_truths.append(row["Targets"])

    predictions = loglikelihoods_to_predictions(log_likelihoods, labels)
    accuracy, f1 = compute_f1_and_accuracy(predictions, ground_truths, labels)
    print(f"Accuracy: {accuracy}, F1 Score: {f1}")


def do_eval_compute(df, labels):
    """
    This function gets the accuracy and F1 score for a task
    given the labels for the task, the logliklihoods and
    targets

    labels: <list> of target labels
    """
    log_likelihoods = df["Log-Likelihood"].astype(float)
    ground_truths = df["Targets"]
    predictions = loglikelihoods_to_predictions(log_likelihoods, labels)
    accuracy, f1 = compute_f1_and_accuracy(predictions, ground_truths, labels)
    return f1


def evaluate_zindi_org(df):
    """
    This function is the same as main function above, it just does
    things in memory so the function can be used for the final eval
    as well.

    First step is to separate df such that each task has its own accuracy calc
    This turns the submissionn file into scores
    Inputs
    df: submission file df

    Outputs:
    Average score accross all the tasks
    """
    scores = []
    df_temp = df[(df["Task"] == "sentiment") & (df["Langs"] == "hausa")]
    labels = ["Kyakkyawa", "Tsaka-tsaki", "Korau"]
    res = do_eval_compute(df_temp, labels)
    scores.append(res)
    print("Score for Sentiment Hausa:", res)

    df_temp = df[(df["Task"] == "sentiment") & (df["Langs"] == "swahili")]
    labels = ["Chanya", "Wastani", "Hasi"]
    res = do_eval_compute(df_temp, labels)
    scores.append(res)
    print("Score for Sentiment Swahili:", res)

    labels = ["0", "1", "2"]
    df_temp = df[(df["Task"] == "xnli") & (df["Langs"] == "hau")]
    res = do_eval_compute(df_temp, labels)
    scores.append(res)
    print("Score for AfriXnli Hausa:", res)

    df_temp = df[(df["Task"] == "xnli") & (df["Langs"] == "swa")]
    res = do_eval_compute(df_temp, labels)
    scores.append(res)
    print("Score for AfriXnli Swahili:", res)

    df_temp = df[df["Task"] == "mmt"]
    res = calculate_chrf(df_temp)
    print("Score for MMT:", res)
    scores.append(res["score"] / 100.0)

    avg_score = mean(float(n) for n in scores if not np.isnan(float(n)))
    print("Average performance score accross tasks and langs: ", avg_score)
    return avg_score


# new functions from here

# def evaluate_zindi(df):
#   """
#   Updated 17 October 2024
#   This function is the same as main function above, it just does
#   things in memory so the function can be used for the final eval
#   as well.

#   First step is to separate df such that each task has its own accuracy calc
#   This turns the submissionn file into scores
#   Inputs
#   df: submission file df

#   Outputs:
#   Average score accross all the tasks
#   """
#   print("it is usung this function")
#   scores = []
#   df_temp = df[(df['Task'] == 'sentiment') & (df['Langs'] == 'hausa')]
#   labels = ["Kyakkyawa","Tsaka-tsaki","Korau"]
#   res = do_eval_compute(df_temp, labels)
#   scores.append(res)
#   print("Score for Sentiment Hausa:", res)

#   df_temp = df[(df['Task'] == 'sentiment') & (df['Langs'] == 'swahili')]
#   labels=["Chanya","Wastani","Hasi"]
#   res = do_eval_compute(df_temp, labels)
#   scores.append(res)
#   print("Score for Sentiment Swahili:", res)

#   labels = ["0","1","2"]
#   df_temp = df[(df['Task'] == 'xnli') & (df['Langs'] == 'hau')]
#   res = do_eval_compute(df_temp, labels)
#   scores.append(res)
#   print("Score for AfriXnli Hausa:", res)

#   df_temp = df[(df['Task'] == 'xnli') & (df['Langs']=='swa')]
#   res = do_eval_compute(df_temp, labels)
#   scores.append(res)
#   print("Score for AfriXnli Swahili:", res)

#   df_temp = df[df['Task'] == 'mmt']
#   res = calculate_chrf(df_temp)
#   print("Score for MMT:", res)
#   scores.append(res['score']/100.0)

#   avg_score = mean(float(n) for n in scores if not np.isnan(float(n)))
#   print("Average performance score accross tasks and langs: ", avg_score)
#   return avg_score


def generate_char_ngrams(text, n):
    """
    Generate character n-grams from the given text. For chrf metric from scratch

    Parameters:
    text (str): Input sentence
    n (int): Length of n-grams

    Returns:
    list: List of character n-grams
    """
    text = text.replace(" ", "")  # Remove spaces
    ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]
    return ngrams


def chrf_metric(reference, hypothesis, n=6):
    """
    Compute the ChrF score between a reference and a hypothesis.

    Parameters:
    reference (str): The reference sentence
    hypothesis (str): The hypothesis sentence
    n (int): Maximum character n-gram length

    Returns:
    float: ChrF score
    """
    # Initialize counts
    total_precision = 0
    total_recall = 0
    total_fscore = 0

    for i in range(1, n + 1):
        # Generate n-grams for both reference and hypothesis
        ref_ngrams = generate_char_ngrams(reference, i)
        hyp_ngrams = generate_char_ngrams(hypothesis, i)

        # Calculate the number of n-gram overlaps
        overlap = 0
        for ngram in hyp_ngrams:
            if ngram in ref_ngrams:
                overlap += 1
                ref_ngrams.remove(ngram)  # Remove to avoid double-counting

        # Precision and recall
        precision = overlap / len(hyp_ngrams) if hyp_ngrams else 0
        recall = overlap / len(ref_ngrams) if ref_ngrams else 0

        # F-score calculation for the current n-gram
        if precision + recall == 0:
            fscore = 0
        else:
            fscore = 2 * (precision * recall) / (precision + recall)

        total_fscore += fscore

    # Return the average F-score across all n-grams
    chrf_score = total_fscore / n
    return chrf_score


def f1_score_from_scratch(true_labels, predicted_labels):
    """
    Calculate the F1 score from scratch
    true_labels: List of true labels (ground truth)
    predicted_labels: List of predicted labels (output of the model)

    Returns:
    F1 score (float)
    """

    # Initialize True Positives (TP), False Positives (FP), False Negatives (FN)
    TP = FP = FN = 0

    for true, pred in zip(true_labels, predicted_labels):
        if true == 1 and pred == 1:
            TP += 1  # True Positive
        elif true == 0 and pred == 1:
            FP += 1  # False Positive
        elif true == 1 and pred == 0:
            FN += 1  # False Negative

    # Avoid division by zero
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    if precision + recall == 0:
        return 0

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1


def loglikelihoods_to_predictions(
    log_likelihoods: List[float], labels: List[str]
) -> List[str]:
    predictions = []

    for likelihoods in log_likelihoods:
        if isinstance(likelihoods, list):
            predicted_label_idx = np.argmax(likelihoods)
            predictions.append(labels[predicted_label_idx])
        else:
            print(f"Invalid likelihood format: {likelihoods}")
    return predictions


def compute_f1_and_accuracy(
    predictions: List[str], ground_truths: List[str], labels: List[str]
) -> (float, float):
    label_to_id = {label: i for i, label in enumerate(labels)}

    y_true = [label_to_id[label] for label in ground_truths]
    y_pred = [label_to_id[label] for label in predictions]

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    return accuracy, f1


def calculate_chrf(df):
    """
    This function claculates CHRF when given a prediction and ground truth for translation

    Inputs:
    df: <pandas dataframe> with prediction and ground truth for machine translation

    Outputs:
    score: <dict> chrf results
    """
    chrf_metric = evaluate.load("chrf")
    references = format_references(df["Targets"].to_list())
    predictions = df["Response"].to_list()
    score = chrf_metric.compute(predictions=predictions, references=references)
    return score


def process_likelihood(likelihood_str: str) -> List[float]:
    # clean the string to remove unwanted characters
    clean_str = (
        likelihood_str.replace("tensor(", "").replace(")", "").strip()
    )  # remove 'tensor(' and ')'
    clean_str = (
        clean_str.replace("[[", "").replace("]]", "").strip()
    )  # remove extra brackets
    clean_str = (
        clean_str.replace(" device='cuda:0'", "")
        .replace(" dtype=torch.float16", "")
        .strip()
    )  # remove device and dtype info
    clean_str = clean_str.replace(
        "tensor", ""
    ).strip()  # remove any instances of 'tensor'

    # remove any empty strings caused by extra commas
    clean_str = clean_str.replace(",,", ",")  # remove duplicate commas if they exist

    # Convert to a list of floats
    likelihood = [
        float(x) for x in clean_str.split(",") if x.strip()
    ]  # ensure non-empty strings are converted
    return likelihood


def evaluate_zindi(csv_file_path):
    log_likelihoods = []
    ground_truths = []

    with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        scores = []
        y_pred_sent = []
        y_true_sent = []
        y_pred_xnli = []
        y_true_xnli = []

        for row in reader:
            if "sent" in row["ID"] or "xnli" in row["ID"]:
                if "sent" in row["ID"] and "swahili" in row["ID"]:
                    labels = ["Chanya", "Wastani", "Hasi"]
                if "sent" in row["ID"] and "hausa" in row["ID"]:
                    labels = ["Kyakkyawa", "Tsaka-tsaki", "Korau"]
                if "xnli" in row["ID"]:
                    labels = ["0", "1", "2"]

                # Use the output of process_likelihood directly
                predicted_label = process_likelihood(row["Response"])
                label_to_id = {label: i for i, label in enumerate(labels)}

                if "xnli" in row["ID"]:
                    y_pred_xnli.append(predicted_label)
                    y_true_xnli.append(label_to_id[row["Targets"]])
                if "sent" in row["ID"]:
                    y_pred_sent.append(predicted_label)
                    y_true_sent.append(label_to_id[row["Targets"]])

            elif "mt" in row["ID"]:
                chrf_pred = row["Response"]
                chrf_true = row["Targets"]
                chrfs = chrF(reference=chrf_true, hypothesis=chrf_pred)
                scores.append(chrfs)

        print("Chrf MT: ", np.mean(scores))

        # F1 score for sentiment
        f1_sent = calculate_f1(np.array(y_true_sent), np.array(y_pred_sent), 3)
        scores.append(f1_sent)
        print("F1score Sentiment: ", f1_sent)

        # F1 score for xnli
        f1_xnli = calculate_f1(np.array(y_true_xnli), np.array(y_pred_xnli), 3)
        scores.append(f1_xnli)
        print("F1score Xnli: ", f1_xnli)

        # Zindi score: Average of all performances
        zindi_score = np.mean(scores)

    return zindi_score


# From scratch implementation of chrf
def get_char_ngrams(sentence, n):
    """Generate character n-grams from a sentence."""
    sentence = sentence.replace(" ", "")  # Remove spaces for chrF
    return [sentence[i : i + n] for i in range(len(sentence) - n + 1)]


def precision_recall(reference, hypothesis, n):
    """Calculate precision and recall for character n-grams."""
    ref_ngrams = get_char_ngrams(reference, n)
    hyp_ngrams = get_char_ngrams(hypothesis, n)

    ref_count = Counter(ref_ngrams)
    hyp_count = Counter(hyp_ngrams)

    common_ngrams = ref_count & hyp_count
    true_positives = sum(common_ngrams.values())

    precision = true_positives / max(len(hyp_ngrams), 1)
    recall = true_positives / max(len(ref_ngrams), 1)

    return precision, recall


def f_score(precision, recall, beta=1):
    """Calculate the F1 score."""
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)


def chrF(reference, hypothesis, max_n=6, beta=2):
    """Calculate the chrF score from scratch."""
    precisions = []
    recalls = []

    for n in range(1, max_n + 1):
        precision, recall = precision_recall(reference, hypothesis, n)
        precisions.append(precision)
        recalls.append(recall)

    avg_precision = sum(precisions) / max_n
    avg_recall = sum(recalls) / max_n

    return f_score(avg_precision, avg_recall, beta)


# From scratch implementation f1score 3 class
def calculate_f1(true_labels, pred_labels, num_classes):
    f1_scores = []

    for i in range(num_classes):
        TP = np.sum((true_labels == i) & (pred_labels == i))  # True Positives
        FP = np.sum((true_labels != i) & (pred_labels == i))  # False Positives
        FN = np.sum((true_labels == i) & (pred_labels != i))  # False Negatives

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        # Calculate F1 score
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        f1_scores.append(f1)

    macro_f1 = np.mean(f1_scores)

    return macro_f1
