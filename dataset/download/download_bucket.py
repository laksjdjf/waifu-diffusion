from inspect import trace
import os
import json
import requests
import multiprocessing
import tqdm
import random
import re
from concurrent import futures
import io
import tarfile
import glob

from PIL import Image, ImageOps

import torch
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from aesthetic import image_embeddings_image, Classifier

import argparse
import numpy as np

import pandas as pd
from Utils import dbimutils

import cv2

from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str, required=False, default='links.json')
parser.add_argument('--out_file', '-o', type=str, required=False, default='datasets')
parser.add_argument('--threads', '-p', required=False, default=16, type=int)
parser.add_argument('--start', '-s', required=False, default=0, type=int)
parser.add_argument('--end', '-e', required=False, type=int)
parser.add_argument('--resolution', '-r', required=False, default=768, type=int,help="bucketの解像度")
parser.add_argument('--min_length', required=False, default=512, type=int)
parser.add_argument('--max_length', required=False, default=1024, type=int)
parser.add_argument('--max_ratio', required=False, default=2.0, type=float)
parser.add_argument('--aes_threshold', required=False, default=0.9, type=float,help="aesthetic scoreの閾値")
parser.add_argument('--tag_threshold', required=False, default=-1, type=float,help="設定非推奨")
args = parser.parse_args()

#this code is modified from download.py
#added bucketing and aesthetic selection.
class DownloadManager():
    def __init__(self, max_threads: int = 32):
        self.failed_downloads = []
        self.max_threads = max_threads
        self.uuid = args.out_file
        self.buckets, self.ratios = self.make_buckets()
        self.dic = {}
        if args.aes_threshold > 0:
            self.clipprocessor, self.clipmodel, self.aes_model = self.load_aesthetic_model()
        if args.tag_threshold > 0:
            self.model = load_model("networks/ViTB16_11_03_2022_07h05m53s")
            self.label_names = pd.read_csv("2022_0000_0899_6549/selected_tags.csv")
    
    # args = (post_id, link, caption_data)
    def download(self, args_thread):
        try:
            image = Image.open(requests.get(args_thread[1], stream=True).raw).convert('RGB')
            image = self.resize_image(image)
            
            if args.aes_threshold > 0:
                pred = self.pred_aesthetic(image)
                if pred < args.aes_threshold:
                    return
            
                                                   
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='PNG')
            __key__ = '%07d' % int(args_thread[0])
            
            if str(image.size) not in self.dic:
                self.dic[str(image.size)] = [__key__]
            else:
                self.dic[str(image.size)].append(__key__)            
            
            
            image = image_bytes.getvalue()
            with open(f'{self.uuid}/{__key__}.png', 'wb') as f:
                f.write(image)
            
            #とりあえず保留
            if args.tag_threshold > 0:
                img = dbimutils.smart_imread(f'{self.uuid}/{__key__}.png')
                img = dbimutils.smart_24bit(img)
                img = dbimutils.make_square(img, 448)
                img = dbimutils.smart_resize(img, 448)
                img = img.astype(np.float32)
                img = np.expand_dims(img, 0)

                probs = self.model(img, training=False)
                self.label_names["probs"] = probs[0]

                # Everything else is tags: pick any where prediction confidence > threshold
                tags_names = self.label_names[4:]
                found_tags = tags_names[tags_names["probs"] > args.tag_threshold][["name"]]
                args_thread[2]["tagger"] = " ".join(list(found_tags["name"]))
            
            caption = str(json.dumps(args_thread[2]))
            with open(f'{self.uuid}/{__key__}.txt', 'w') as f:
                f.write(caption)

        except Exception as e:
            import traceback
            print(e, traceback.print_exc())
            self.failed_downloads.append((args_thread[0], args_thread[1], args_thread[2]))
    
    def download_urls(self, file_path):
        with open(file_path) as f:
            data = json.load(f)
            keys = list(data.keys())
            
        end_id = args.end if args.end is not None else len(keys)
        data = {keys[i]:data[keys[i]] for i in range(args.start,end_id)}
        
        thread_args = []

        delimiter = '\\' if os.name == 'nt' else '/'
        
        if not os.path.exists(f'{self.uuid}'):
            os.mkdir(f'{self.uuid}')

        print(f'Loading {file_path} for downloading on {self.max_threads} threads... Writing to dataset {self.uuid}')

        # create initial thread_args
        for k, v in tqdm.tqdm(data.items()):
            thread_args.append((k, v['file_url'], v))
        
        # divide thread_args into chunks divisible by max_threads
        chunks = []
        for i in range(0, len(thread_args), self.max_threads):
            chunks.append(thread_args[i:i+self.max_threads])
        
        print(f'Downloading {len(thread_args)} images...')

        # download chunks synchronously
        for chunk in tqdm.tqdm(chunks):
            with futures.ThreadPoolExecutor(args.threads) as p:
                p.map(self.download, chunk)
                
        with open(f'{self.uuid}/buckets.json',"w") as f:
            f.write(json.dumps(self.dic))

        if len(self.failed_downloads) > 0:
            print("Failed downloads:")
            for i in self.failed_downloads:
                print(i[0])
            print("\n")
            
    def make_buckets(self):
        increment = 64
        max_pixels = args.resolution*args.resolution

        buckets = set()
        buckets.add((args.resolution, args.resolution))

        width = args.min_length
        while width <= args.max_length:
            height = min(args.max_length, (max_pixels // width ) - (max_pixels // width ) % increment)
            ratio = width/height
            if 1/args.max_ratio <= ratio <= args.max_ratio:
                buckets.add((width, height))
                buckets.add((height, width))
            width += increment
        buckets = list(buckets)
        ratios = [w/h for w,h in buckets]
        buckets = np.array(buckets)[np.argsort(ratios)]
        ratios = np.sort(ratios)
        return buckets, ratios
    
    def resize_image(self, image :Image):
        image = image.convert("RGB")
        ratio = image.width / image.height
        ar_errors = self.ratios - ratio
        indice = np.argmin(np.abs(ar_errors))
        bucket_width, bucket_height = self.buckets[indice]
        ar_error = ar_errors[indice]
        if ar_error <= 0:
            temp_width = int(image.width*bucket_height/image.height)
            image = image.resize((temp_width,bucket_height))
            left = (temp_width - bucket_width) / 2
            right = bucket_width + left
            image = image.crop((left,0,right,bucket_height))
        else:
            temp_height = int(image.height*bucket_width/image.width)
            image = image.resize((temp_height,bucket_width))
            upper = (temp_height - bucket_height) / 2
            lower = bucket_height + upper
            image = image.crop((0,upper,bucket_width,lower))
        return image
    
    def load_aesthetic_model(self):
        aesthetic_path = 'aes-B32-v0.pth'
        clip_name = 'openai/clip-vit-base-patch32'

        clipprocessor = CLIPProcessor.from_pretrained(clip_name)
        clipmodel = CLIPModel.from_pretrained(clip_name).to('cuda').eval()

        aes_model = Classifier(512, 256, 1).to("cuda")
        aes_model.load_state_dict(torch.load(aesthetic_path))

        return clipprocessor, clipmodel, aes_model
    
    def pred_aesthetic(self, image):
        image_embeds = image_embeddings_image(image, self.clipmodel, self.clipprocessor)
        prediction = self.aes_model(torch.from_numpy(image_embeds).float().to('cuda'))
        return prediction.item()

        
if __name__ == '__main__':
    dm = DownloadManager(max_threads=args.threads)
    dm.download_urls(args.file)
