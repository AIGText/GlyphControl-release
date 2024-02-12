"""Calculates the CLIP Scores

The CLIP model is a contrasitively learned language-image model. There is
an image encoder and a text encoder. It is believed that the CLIP model could 
measure the similarity of cross modalities. Please find more information from 
https://github.com/openai/CLIP.

The CLIP Score measures the Cosine Similarity between two embedded features.
This repository utilizes the pretrained CLIP Model to calculate 
the mean average of cosine similarities. 
"""

import os
from PIL import Image
import clip
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from argparse import ArgumentParser

IMAGE_EXTENSIONS = 'jpg'
PROMPT_TYPE = {'Sign', 'GlyphDraw'} # "Sign": SimpleBench; "GlyphDraw": CreativeBench

def parser_fn():

    parser = ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size to use')
    parser.add_argument('--clip-model', type=str, default='ViT-B/32',
                        help='CLIP model to use')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of processes')
    parser.add_argument('--prompt_type', type=str, default=None, 
                        help='Sign or GlyphDraw')
    parser.add_argument('--img_path', type=str, 
                        help='Image folder path')
    parser.add_argument('--img_path_multi', type=str, default=None,
                    help='The path including multiple Image folder paths')
    parser.add_argument('--ckpt_name', type=str, default=None,
                    help='The checkpoint name')    
    return parser

class DummyDataset(Dataset):
    
    def __init__(self, img_path, prompt_type, 
                 transform = None,
                 tokenizer = None) -> None:
        super().__init__()
        
        if prompt_type is None:
            if "GlyphDraw" in img_path:
                prompt_type = 'GlyphDraw'
            else:
                prompt_type = 'Sign'
        
        assert prompt_type in PROMPT_TYPE
        self.img_path = img_path
        self.prompt_type = prompt_type
        self.transform = transform
        self.tokenizer = tokenizer
        if prompt_type == 'Sign':
            self._prepare_sign(img_path)
        if prompt_type == 'GlyphDraw':
            self._prepare_glyphdraw(img_path)
        print(f"{len(self.img_path_list)} image paths")
        print(f"First example:\n Image Path: {self.img_path_list[:1]}\n Prompt:{self.text_list[:1]}")
        
        assert len(self.img_path_list) == len(self.text_list)
        
    def _prepare_sign(self, img_path):
        
        self.img_path_list = []
        self.text_list = []
        
        for item in [i for i in os.listdir(img_path) if "." not in i]:
            path = os.path.join(img_path, item)
            for sub_item in [i for i in os.listdir(path) if IMAGE_EXTENSIONS in i and item in i and "glyph" not in i]:
                sub_path = os.path.join(path, sub_item)
                
                self.img_path_list.append(sub_path)
                self.text_list.append(f'A sign that says "{item}"')
            
    def _prepare_glyphdraw(self, img_path):
        
        self.img_path_list = []
        self.text_list = []
        
        for item in [i for i in os.listdir(img_path) if "." not in i]:
            path = os.path.join(img_path, item)
            
            with open(os.path.join(path, "prompt.txt"), 'r') as fp:
                prompt = fp.readline()
            
            for sub_item in [i for i in os.listdir(path) if IMAGE_EXTENSIONS in i and item in i and "glyph" not in i]:
                sub_path = os.path.join(path, sub_item)
                
                self.img_path_list.append(sub_path)
                self.text_list.append(prompt)
            
    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        
        img_path = self.img_path_list[index]
        text = self.text_list[index]
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        if self.tokenizer:
            text = self.tokenizer(text).squeeze()
            
        return image, text

@torch.no_grad()
def calculate_clip_score(dataloader, model, device):
    score_acc = 0.
    sample_num = 0.
    logit_scale = model.logit_scale.exp()
    print(f"Clip Model logit_scale is:{logit_scale}")
    for image, text in dataloader:
        
        image_features = model.encode_image(image.to(device))

        text_features = model.encode_text(text.to(device))
        
        # normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True).to(torch.float32)
        text_features = text_features / text_features.norm(dim=1, keepdim=True).to(torch.float32)
        
        # calculate scores
        score = logit_scale * (image_features * text_features).sum()
        score_acc += score
        sample_num += image.shape[0]
    
    return score_acc / sample_num

def main(img_path, args):

    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    print(device)
    num_workers = args.num_workers
    print("--------------------------")
    print(f"Evaluating on the {img_path}")
    print('Loading CLIP model: {}'.format(args.clip_model))
    model, preprocess = clip.load(args.clip_model, device=device)
    
    dataset = DummyDataset(img_path, args.prompt_type,
                           transform=preprocess, tokenizer=clip.tokenize)
    
    if len(dataset) != 400:
        return
    dataloader = DataLoader(dataset, args.batch_size, 
                            num_workers=num_workers, pin_memory=True)
    dataloader = tqdm(dataloader)
    
    print('Calculating CLIP Score:')
    clip_score = calculate_clip_score(dataloader, model, device)
    clip_score = clip_score.cpu().item()
    print('CLIP Score: ', clip_score)


if __name__ == '__main__':

    args = parser_fn().parse_args()
    if args.img_path_multi is not None:
        from glob import glob
        img_paths = glob(args.img_path_multi + "/*")
        for img_path in img_paths:
            if not os.path.isdir(img_path):
                print(img_path, "is not a directory")
                continue
            if args.ckpt_name is not None:
                img_path = os.path.join(img_path, args.ckpt_name)
            main(img_path, args)
    else:
        main(args.img_path, args)