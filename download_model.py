import os
import gdown

MODEL_PATH = "model_quantized.pt"
FILE_ID = "13VRKr_0oh0TLMTqNNantF0qQaq7M_WIh"   # 🔥 Your file ID here
URL = f"https://drive.google.com/uc?export=download&id=13VRKr_0oh0TLMTqNNantF0qQaq7M_WIh"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Downloading...")
        gdown.download(URL, MODEL_PATH, quiet=False)
        print("Model downloaded successfully.")
    else:
        print("Model already exists. Skipping download.")





