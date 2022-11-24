import os 
import numpy as np
import torch
from torchvision import transforms
from diffusers import AutoencoderKL
from tqdm import tqdm
import argparse
from PIL import Image



parser = argparse.ArgumentParser()
parser.add_argument('--directory', '-d', type=str, required=True)
parser.add_argument('--output_path', '-o', type=str, required=True)
parser.add_argument('--start', '-s', required=False, default=0, type=int)
parser.add_argument('--end', '-e', required=False, type=int)
parser.add_argument('--model','-m' , required=True, type=str)
args = parser.parse_args()

def main():
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae")
    vae.eval()
    vae.to("cuda", dtype=torch.float16)
    
    to_tensor_norm = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    path = args.directory.rstrip("/") + "/"
    output_path = args.output_path.rstrip("/") + "/"
    
    files = os.listdir(args.directory)
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    end_id = args.end if args.end is not None else len(files)
    
    for file in tqdm(files[args.start:end_id]):
        if "jpg" not in file:
            continue
        image = Image.open(path + file)
        image_tensor = to_tensor_norm(image).to("cuda",torch.float16)
        image_tensors = torch.stack([image_tensor]) #batch size 1のごみ実装
        with torch.no_grad():
            latent = vae.encode(image_tensors).latent_dist.sample().float().to("cpu").numpy()[0]
        np.save(output_path + file[:-4] + ".npy",latent)

if __name__ == "__main__":
    main()
