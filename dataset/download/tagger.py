import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from Utils import dbimutils

from tensorflow.keras.models import load_model

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--directory', '-d', type=str, required=True)
parser.add_argument('--output_path', '-o', type=str, required=True)
parser.add_argument('--start', '-s', required=False, default=0, type=int)
parser.add_argument('--end', '-e', required=False, type=int)
parser.add_argument('--image_size', '-i', required=False, default=448, type=int)
parser.add_argument('--batch_size', '-b', required=False, default=64, type=int)
parser.add_argument('--threshold', '-t', required=False, default=0.35, type=float)
args = parser.parse_args()

#WD 1.4 tagger
def main():
    model = load_model("networks/ViTB16_11_03_2022_07h05m53s")
    label_names = pd.read_csv("2022_0000_0899_6549/selected_tags.csv")
    
    path = args.directory.rstrip("/") + "/"
    output_path = args.output_path.rstrip("/") + "/"
    files = os.listdir(path)
    end_id = args.end if args.end is not None else len(files)
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    imgs = []
    batch_keys = []
    for i in tqdm(range(args.start,end_id)):
        file = files[i]
        if "png" not in file:
            continue
        img = dbimutils.smart_imread(path + file)
        img = dbimutils.smart_24bit(img)
        img = dbimutils.make_square(img, args.image_size)
        img = dbimutils.smart_resize(img, args.image_size)
        img = img.astype(np.float32)
        imgs.append(img)
        batch_keys.append(file[:-4])
        
        if len(imgs) == args.batch_size:
            probs = model(np.array(imgs), training=False)
            for j in range(len(imgs)):
                label_names["probs"] = probs[j]
                tags_names = label_names[4:]
                found_tags = tags_names[tags_names["probs"] > args.threshold][["name"]]
                tags = " ".join(list(found_tags["name"]))
                with open(path + batch_keys[j] + ".txt","r") as f:
                    caption = f.read()
                with open(output_path + batch_keys[j] + ".txt","w") as f:
                    f.write(caption[:-1] + ', "tagger": "' + tags + '"}')
            imgs = []
            batch_keys = []
    
    probs = model(np.array(imgs), training=False)
    for j in range(len(imgs)):
        label_names["probs"] = probs[j]
        tags_names = label_names[4:]
        found_tags = tags_names[tags_names["probs"] > args.threshold][["name"]]
        tags = " ".join(list(found_tags["name"]))
        with open(path + batch_keys[j] + ".txt","r") as f:
            caption = f.read()
        with open(output_path + batch_keys[j] + ".txt","w") as f:
            f.write(caption[:-1] + ', "tagger": "' + tags + '"}')
    imgs = []
    batch_keys = []

if __name__ == "__main__":
    main()
