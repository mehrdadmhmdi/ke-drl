import os
import urllib.request
import torch

def get_dataset(fname="expedia_train_rl.csv"):
    local = os.path.join(os.path.dirname(__file__), "example", "Expedia_data", fname)
    if not os.path.exists(local):
        url = (
            "https://raw.githubusercontent.com/mehrdadmhmdi/ke-drl/"
            "main/example/Expedia_data/" + fname
        )
        print("Downloading data from", url)
        urllib.request.urlretrieve(url, local)
    return local  # or return reading logic, e.g., pd.read_csv(local)
