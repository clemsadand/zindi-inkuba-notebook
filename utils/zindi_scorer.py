import csv
from collections import Counter
from typing import List  # this could be removed from the code if necessary

import numpy as np


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
    PARAM_SIZE = 421939200.0  # the size of Inkuba

    with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        count = 0
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

                likelihoods = process_likelihood(row["Response"])
                label_to_id = {label: i for i, label in enumerate(labels)}

                if "xnli" in row["ID"]:
                    y_pred_xnli.append(np.argmax(likelihoods))
                    y_true_xnli.append(label_to_id[row["targets"]])
                if "sent" in row["ID"]:
                    y_pred_sent.append(np.argmax(likelihoods))
                    y_true_sent.append(label_to_id[row["targets"]])

            elif "mt" in row["ID"]:
                chrf_pred = row["Response"]
                chrf_true = row["targets"]
                chrfs = chrF(reference=chrf_true, hypothesis=chrf_pred)
                scores.append((chrfs))
            elif "size" in row["ID"]:
                size = int(row["Response"])
        print("Chrf MT: ", np.mean(scores))
        # f1 score for sentiment
        f1_sent = calculate_f1(np.array(y_true_sent), np.array(y_pred_sent), 3)
        scores.append(f1_sent)
        print("F1score Sentiment: ", f1_sent)
        # f1 score for xnli
        f1_xnli = calculate_f1(np.array(y_true_xnli), np.array(y_pred_xnli), 3)
        scores.append(f1_xnli)
        print("F1score Xnli: ", f1_xnli)

        average_score = np.sum(scores) / len(scores)
        # Zindi score takes the average of all perfromances (out of 1)
        # It scales this value by the model size making the total possible score out of 2
        # ie if the model is 100x smaller than Inkuba then they can double their score
        # We then divide by 2 to get the score to be out of 1 again
        zindi_score = (average_score + (1 - (size / PARAM_SIZE)) * average_score) / 2
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
