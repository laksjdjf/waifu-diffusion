import json

import argparse
import shutil
import numpy as np

import torch
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from aesthetic import image_embeddings_image, Classifier
import os
from tqdm import tqdm

import math
import requests

from PIL import Image
import cv2

import random
import re

from diffusers import AutoencoderKL

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process
from itertools import islice

import time

parser = argparse.ArgumentParser()
parser.add_argument('--json_path', '-j', type=str, default='links.json')
parser.add_argument('--output_path', '-o', type=str, default='test')
parser.add_argument('--threshold', type=int, default=0.9)
parser.add_argument('--bucket_side_min', type=int, default=256, help='The minimum side length of a bucket.')
parser.add_argument('--bucket_side_max', type=int, default=1024, help='The maximum side length of a bucket.')
parser.add_argument('--resolution', type=int, default=768, help='Image resolution to train against. Lower res images will be scaled up to this resolution and higher res images will be scaled down.')
parser.add_argument('--max_ratio', type=float, default=2.0, help='The maximum aspect ratio of bucket (the reciprocal is minimum)')
parser.add_argument("--model_name_or_path", "-m", type=str, help="diffusers path")
parser.add_argument("--start", "-s", type=int, help="start")
parser.add_argument("--goal", "-g", type=int, help="start")

args = parser.parse_args()


def load_aesthetic_model():
    aesthetic_path = 'aes-B32-v0.pth'
    clip_name = 'openai/clip-vit-base-patch32'

    clipprocessor = CLIPProcessor.from_pretrained(clip_name)
    clipmodel = CLIPModel.from_pretrained(clip_name).to('cuda').eval()

    aes_model = Classifier(512, 256, 1).to("cuda")
    aes_model.load_state_dict(torch.load(aesthetic_path))
    
    return clipprocessor, clipmodel, aes_model

def pred_aesthetic(image):
    image_embeds = image_embeddings_image(image, clipmodel, clipprocessor)
    prediction = aes_model(torch.from_numpy(image_embeds).float().to('cuda'))
    return prediction.item()

def make_buckets():
    max_width, max_height = args.resolution,args.resolution
    max_size = args.bucket_side_max
    max_ratio = args.max_ratio
    divisible = 64
    max_area = (max_width // divisible) * (max_height // divisible)

    resos = set()

    size = int(math.sqrt(max_area)) * divisible
    resos.add((size, size))

    size = args.bucket_side_min
    while size <= max_size:
        width = size
        height = min(max_size, (max_area // (width // divisible)) * divisible)
        ratio = width/height
        if 1/max_ratio <= ratio <= max_ratio:
            resos.add((width, height))
            resos.add((height, width))
        size += divisible
    resos = list(resos)
    ratios = [w/h for w,h in resos]
    buckets = np.array(resos)[np.argsort(ratios)]
    buckets = [(int(i),int(j)) for i,j in buckets]
    ratios = [w/h for w,h in buckets]
    return buckets,ratios

def resize_from_buckets(image):
    aspect_ratio = image.width / image.height
    ar_errors = np.array(ratios) - aspect_ratio
    bucket_id = np.abs(ar_errors).argmin()
    reso = buckets[bucket_id]
    ar_error = ar_errors[bucket_id]
    # どのサイズにリサイズするか→トリミングする方向で
    if ar_error <= 0:                   # 横が長い→縦を合わせる
        scale = reso[1] / image.height
    else:
        scale = reso[0] / image.width

    resized_size = (int(image.width * scale + .5), int(image.height * scale + .5))

    # print(image.width, image.height, bucket_id, bucket_resos[bucket_id], ar_errors[bucket_id], resized_size,
    #       bucket_resos[bucket_id][0] - resized_size[0], bucket_resos[bucket_id][1] - resized_size[1])

    assert resized_size[0] == reso[0] or resized_size[1] == reso[
        1], f"internal error, resized size not match: {reso}, {resized_size}, {image.width}, {image.height}"
    assert resized_size[0] >= reso[0] and resized_size[1] >= reso[
        1], f"internal error, resized size too small: {reso}, {resized_size}, {image.width}, {image.height}"

    # 画像をリサイズしてトリミングする
    # PILにinter_areaがないのでcv2で……
    image = np.array(image)
    image = cv2.resize(image, resized_size, interpolation=cv2.INTER_AREA)
    if resized_size[0] > reso[0]:
        trim_size = resized_size[0] - reso[0]
        image = image[:, trim_size//2:trim_size//2 + reso[0]]
    elif resized_size[1] > reso[1]:
        trim_size = resized_size[1] - reso[1]
        image = image[trim_size//2:trim_size//2 + reso[1]]
    assert image.shape[0] == reso[1] and image.shape[1] == reso[0], f"internal error, illegal trimmed size: {image.shape}, {reso}"
    
    return image
    
def get_image_from_url(url):
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    return image

class CaptionProcessor(object):
    def __init__(self, copyright_rate, character_rate, general_rate, artist_rate, normalize, caption_shuffle, transforms, max_size, resize, random_order):
        self.copyright_rate = copyright_rate
        self.character_rate = character_rate
        self.general_rate = general_rate
        self.artist_rate = artist_rate
        self.normalize = normalize
        self.caption_shuffle = caption_shuffle
        self.transforms = transforms
        self.max_size = max_size
        self.resize = resize
        self.random_order = random_order
    
    def clean(self, text: str):
        text = ' '.join(set([i.lstrip('_').rstrip('_') for i in re.sub(r'\([^)]*\)', '', text).split(' ')])).lstrip().rstrip()
        if self.caption_shuffle:
            text = text.split(' ')
            random.shuffle(text)
            text = ' '.join(text)
        if self.normalize:
            text = ', '.join([i.replace('_', ' ') for i in text.split(' ')]).lstrip(', ').rstrip(', ')
        return text

    def get_key(self, val_dict, key, clean_val = True, cond_drop = 0.0, prepend_space = False, append_comma = False):
        space = ' ' if prepend_space else ''
        comma = ',' if append_comma else ''
        if random.random() < cond_drop:
            if (key in val_dict) and val_dict[key]:
                if clean_val:
                    return space + self.clean(val_dict[key]) + comma
                else:
                    return space + val_dict[key] + comma
        return ''

    def __call__(self, sample):
        # preprocess caption
        caption_data = sample
        if not self.random_order:
            character = self.get_key(caption_data, 'tag_string_character', True, self.character_rate, False, True)
            copyright = self.get_key(caption_data, 'tag_string_copyright', True, self.copyright_rate, True, True)
            artist = self.get_key(caption_data, 'tag_string_artist', True, self.artist_rate, True, True)
            general = self.get_key(caption_data, 'tag_string_general', True, self.general_rate, True, False)
            tag_str = f'{character}{copyright}{artist}{general}'.lstrip().rstrip(',')
        else:
            character = self.get_key(caption_data, 'tag_string_character', False, self.character_rate, False)
            copyright = self.get_key(caption_data, 'tag_string_copyright', False, self.copyright_rate, True, False)
            artist = self.get_key(caption_data, 'tag_string_artist', False, self.artist_rate, True, False)
            general = self.get_key(caption_data, 'tag_string_general', False, self.general_rate, True, False)
            tag_str = self.clean(f'{character}{copyright}{artist}{general}').lstrip().rstrip(' ')
        sample = tag_str

        return sample

def step(key, dic):
    try:
        image = get_image_from_url(dic["file_url"])
    except:
        print("failed download ",key)
        return 0
    image = resize_from_buckets(image)
    pred = pred_aesthetic(image)

    if pred < 0.9:
        return 0
    
    image_tensor = to_tensor_norm(image).to("cuda",torch.float16)
    image_tensors = torch.stack([image_tensor]) #batch size 1のごみ実装
    with torch.no_grad():
        latent = vae.encode(image_tensors).latent_dist.sample().float().to("cpu").numpy()[0]
    caption = processor(dic)
    
    np.savez(os.path.join(args.output_path, key +".npz"),latent)
    with open(os.path.join(args.output_path, key +".txt"), 'w') as f:
        f.write(caption)
    
    return 1



def run(data):            
    with ProcessPoolExecutor(6) as e:
        results = list(tqdm(e.map(step, data.keys(),data.values()),total=len(data)))
    return results

def all_step(data):
    sum_count = 0
    for k,d in tqdm(data.items()):
        sum_count += step(k,d)
    return sum_count

if __name__ == '__main__':
    #global 
    with open(args.json_path) as f:
        data = json.load(f)
    clipprocessor, clipmodel, aes_model = load_aesthetic_model()
    buckets, ratios = make_buckets()
    print(buckets)
    vae = AutoencoderKL.from_pretrained(args.model_name_or_path, subfolder="vae")
    vae.eval()
    vae.to("cuda", dtype=torch.float16)
    
    to_tensor_norm = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    processor = CaptionProcessor(1.0, 1.0, 1.0, 1.0, True, False, None, 768, False, False)
    
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    
    keys = list(data.keys())
    if args.start is not None:
        data = {keys[i]:data[keys[i]] for i in range(args.start,args.goal)}
    all_step(data)
