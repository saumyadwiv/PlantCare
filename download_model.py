import gdown

url = 'https://drive.google.com/file/d/1GWJ5HC8LxQsExmHEZWf7q89MRUquqt7U/view?usp=drive_link'  # replace with your file ID
output = 'plant_disease_model_1_latest.pt'

gdown.download(url, output, quiet=False)
