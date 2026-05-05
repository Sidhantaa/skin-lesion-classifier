import os
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "nodoubttome/skin-cancer9-classesisic"
DOWNLOAD_PATH = "data"

# Download data raw data from Kaggle
def download_dataset():
    os.makedirs(DOWNLOAD_PATH, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    try:
        print("Staring download.")
        api.dataset_download_files(DATASET,path=DOWNLOAD_PATH,unzip=True)
        print("Dataset downloaded successfully.")

    except Exception as e:
        print("Failed to download dataset.")
        print(str(e))

# Move the Train and Test folders up 1 level so 'data' is the top level folder
def flatten_folder(base_path="data"):
    isic_path = os.path.join(base_path, "Skin cancer ISIC The International Skin Imaging Collaboration")

    if not os.path.exists(isic_path):
        print("No ISIC folder found. Skipping flatten.")
        return

    for item in os.listdir(isic_path):
        src = os.path.join(isic_path, item)
        dst = os.path.join(base_path, item)

        if os.path.exists(dst):
            print(f"Warning: {dst} already exists. Skipping {item}.")
            continue

        shutil.move(src, dst)

    os.rmdir(isic_path)

    print("Flattened ISIC folder structure.")

if __name__ == "__main__":
    download_dataset()
    flatten_folder()