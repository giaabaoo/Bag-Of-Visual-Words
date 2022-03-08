from cProfile import label
import os 
from pathlib import Path
import json

if __name__ == '__main__':
    label_path = "./data/labels"
    img_path = "./data/images"
    data = {}

    Path("./preprocessed_data/").mkdir(parents=True, exist_ok=True)

    for file in os.listdir(label_path):
        if "query" in file:
            with open(os.path.join(label_path, file), 'r') as f:
                image_name = f.readline().split()[0].replace("oxc1_", "")
                data[image_name] = os.path.join(img_path, image_name) + ".jpg"

    with open(os.path.join("./preprocessed_data", "queries.json"), "w") as f:
        json.dump(data, f, indent=4)
    