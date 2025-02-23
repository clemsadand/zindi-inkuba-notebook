def process_likelihood(likelihood_str: str) -> List[float]:
    """
    Process a likelihood string to clean and convert it to a list of floats.
    """
    # Clean the string to remove unwanted characters
    clean_str = (
        likelihood_str.replace("tensor(", "").replace(")", "").strip()
        .replace("[[", "").replace("]]", "").strip()
        .replace(" device='cuda:0'", "").replace(" dtype=torch.float16", "")
        .replace(" dtype=torch.float32", "").strip()
        .replace("tensor", "").strip()
    )

    # Remove any empty strings caused by extra commas
    clean_str = clean_str.replace(",,", ",")  # Remove duplicate commas if they exist

    # Convert to a list of floats
    likelihood = [
        float(x) for x in clean_str.split(",") if x.strip()
    ]  # Ensure non-empty strings are converted
    return likelihood

def create_submission(output_path, test_flag: bool):
    """
    Creates submission files based on the provided test_flag.

    Args:
    test_flag (bool): If True, creates a test submission file; otherwise, creates a final submission file.
    """
    if test_flag:
        try:
            df1 = pd.read_csv(os.path.join(
                output_path,
                "hau_sent_prediction_dev.csv")
                 )
            df2 = pd.read_csv(os.path.join(
                output_path,
                "swa_sent_prediction_dev.csv")
            )
            df3 = pd.read_csv(os.path.join(
                output_path,
                "hau_mt_prediction_dev.csv")
                             )
            df4 = pd.read_csv(os.path.join(
                output_path,
                "swa_mt_prediction_dev.csv"))
            df5 = pd.read_csv(os.path.join(
                output_path,
                "hau_xnli_prediction_dev.csv"))
            df6 = pd.read_csv(os.path.join(
                output_path,
                "swa_xnli_prediction_dev.csv"))
        except FileNotFoundError as e:
            print(
                "Seems you have not completed all the tasks, please complete all the tasks before attempting to create your submission file"
            )
            raise e
    else:
        filename = "submission.csv"
        try:
            df1 = pd.read_csv(os.path.join(
                output_path,
                "hau_sent_prediction.csv"))
            df2 = pd.read_csv(os.path.join(
                output_path,
                "swa_sent_prediction.csv"))
            df3 = pd.read_csv(os.path.join(
                output_path,
                "hau_mt_prediction.csv"))
            df4 = pd.read_csv(os.path.join(
                output_path,
                "swa_mt_prediction.csv"))
            df5 = pd.read_csv(os.path.join(
                output_path,
                "hau_xnli_prediction.csv"))
            df6 = pd.read_csv(os.path.join(
                output_path,
                "swa_xnli_prediction.csv"))
        except FileNotFoundError as e:
            print(
                "Seems you have not completed all the tasks, please complete all the tasks before attempting to create your submission file"
            )
            raise e

    # Combine and process data
    resmt = pd.concat([df3, df4], ignore_index=True)
    res_log = pd.concat([df1, df2, df5, df6], ignore_index=True)
    res_log.drop(columns=["Response"], inplace=True)
    res_log.rename(columns={"Log-Likelihood": "Response"}, inplace=True)
    res = pd.concat([res_log, resmt], ignore_index=True)

    def process_row(row):
        if "xnli" in row["ID"] or "sent" in row["ID"]:
            likelihoods = process_likelihood(row["Response"])
            predicted_label = np.argmax(likelihoods)
            return predicted_label
        return row["Response"]  # Default for other cases

    # Update the Response column in-place
    res["Response"] = res.apply(process_row, axis=1)

    if test_flag:
        filename = os.path.join(
                output_path,
                "submission_test.csv")
        # Save the submission file
        submission = res[["ID", "Response", "Targets"]]
        submission.to_csv(filename, index=False)
    else:
        filename = os.path.join(
                output_path,
                "submission.csv")
        # Save the submission file
        submission = res[["ID", "Response"]]
        submission.to_csv(filename, index=False)
    return submission
