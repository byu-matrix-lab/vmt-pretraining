import os
import torch
import torchvision
torchvision.set_video_backend("video_reader")

import h5py
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

dataset = h5py.File('data/dataset.h5', 'w')

for video_file in tqdm(os.listdir('data/raw'), desc='Computing CLIP embeddings'):
    path = 'data/raw/' + video_file
    vid = video_file[:-4]

    video = torchvision.io.VideoReader(path, "video")

    frames = []
    for data in video:
        cur_frame = data['data']
        cur_time = data['pts']
        if len(frames) <= int(cur_time*5):
            frames.append(cur_frame)
    
    inputs = processor(text=['a photo of a cat', 'a man runs a race'], images=frames, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

        embeds = outputs.image_embeds.numpy()

    dataset.create_dataset(vid, data=embeds)

dataset.close()
