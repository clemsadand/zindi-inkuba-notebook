# https://huggingface.co/datasets/lelapa/SentimentTrain
#Load dataset from hugging face

from datasets import load_dataset
import pandas as pd

def create_dataset(data_path, dataset_name):
  # Load dataset from Hugging Face
  print("Downloading from ", os.path.join("lelapa", dataset_name))
  ds = load_dataset(os.path.join("lelapa", dataset_name))

  # Convert to DataFrame (assuming "train" split exists)
  df = pd.DataFrame(ds["train"])

  # Save as CSV
  df.to_csv(os.path.join(data_path, dataset_name + ".csv"), index=False)

  print("Downloaded and saved at ",os.path.join(data_path, dataset_name + ".csv"))


for dataset_name in ["Sentiment", "XNLI", "MT"]:
  create_dataset(data_path, dataset_name + "Train")
  create_dataset(data_path, dataset_name + "Test")
  print()
