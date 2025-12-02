import os
import gdown

MODEL_PATH = "plant_disease_model_1_latest.pt"
FILE_ID = "1GWJ5HC8LxQsExmHEZWf7q89MRUquqt7U"   # 🔥 Your file ID here
URL = f"https://drive.google.com/uc?export=download&id=1GWJ5HC8LxQsExmHEZWf7q89MRUquqt7U"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Downloading...")
        gdown.download(URL, MODEL_PATH, quiet=False)
        print("Model downloaded successfully.")
    else:
        print("Model already exists. Skipping download.")



