from typing import Dict
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from einops import rearrange
from ldm.util import instantiate_from_config
# from datasets import load_dataset
import os
from collections import defaultdict
import cv2 
import albumentations
import random
from ldm.data.util import new_process_im, imagenet_process_im
import re

class TextCapsCLDataset(Dataset):
    def __init__(self,
        img_folder,
        caption_file,
        ocr_file,

        no_hint = False, 
        hint_folder = None,
        control_key = "hint",

        image_transforms=[],
        first_stage_key = "jpg", cond_stage_key = "txt",
        OneCapPerImage = False,
        default_caption="",
        ext="jpg",
        postprocess=None,
        return_paths=False,

        filter_data=False,
        filter_words=["sign", "poster"], 
        filter_token_num = False,
        max_token_num = 3,
        
        imagenet_proc = False,
        imagenet_proc_config = None,
        do_new_proc = True,
        new_proc_config = None,
        new_ocr_info = True,

        rendered_txt_in_caption = False,
        caption_choices = ["original", "w_rend_text", "wo_rend_text"],
        caption_drop_rates = [0.1, 0.5, 0.1],

        ) -> None:
        self.root_dir = Path(img_folder)
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key

        # postprocess
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess

        # image transform
        self.imagenet_proc = imagenet_proc
        self.do_new_proc = do_new_proc
        # [Recommend] Please set do_new_proc as True 
        # new_proc_func will crop the original images to maintain the maximum OCR info 
        # and crop out the same area on the hint glyph images.
        if self.do_new_proc:  
            if new_proc_config is not None:
                self.new_proc_func = instantiate_from_config(new_proc_config)
            else:
                self.new_proc_func = new_process_im()
        elif self.imagenet_proc: # ImageNet-type image preprocessing
            if imagenet_proc_config is not None:
                self.imagenet_proc_func = instantiate_from_config(imagenet_proc_config)
            else:
                self.imagenet_proc_func = imagenet_process_im()
            self.process_im = self.imagenet_proc_func 
        else:
            if isinstance(image_transforms, ListConfig):
                image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
            image_transforms.extend([transforms.ToTensor(), # to be checked
                                    transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
            image_transforms = transforms.Compose(image_transforms)
            self.tform = image_transforms
            self.process_im = self.simple_process_im

        # caption
        assert caption_file is not None
        with open(caption_file, "rt") as f:
            ext = Path(caption_file).suffix.lower()
            if ext == ".json":
                captions = json.load(f)
            else:
                raise ValueError(f"Unrecognised format: {ext}")
        self.captions = captions["data"]
        if OneCapPerImage and ocr_file is None:
            new_captions = []
            taken_images = []
            for caption_data in self.captions:
                if caption_data["image_id"] in taken_images:
                    continue
                else:
                    new_captions.append(caption_data)
                    taken_images.append(caption_data["image_id"])
            self.captions = new_captions

        # ocr info
        assert ocr_file is not None
        self.ocr_file = ocr_file
        with open(ocr_file, "r") as f:
            ocrs = json.loads(f.read())
            ocr_data = ocrs['data']
        self.ocr_data = ocr_data

        # hint 
        self.no_hint = no_hint
        self.control_key = control_key
        self.hint_folder = None
        if not self.no_hint:
            if hint_folder is None:
                print("Warning: The folder of hint images is not provided! No hint will be used")
                self.no_hint = True
            else:
                self.hint_folder = Path(hint_folder)


        self.default_caption = default_caption
        self.return_paths = return_paths
        self.filter_data = filter_data
        self.filter_words = filter_words
        self.new_ocr_info = new_ocr_info
        self.rendered_txt_in_caption = rendered_txt_in_caption
        self.filter_token_num = filter_token_num
        self.max_token_num = max_token_num
        self.caption_choices = caption_choices
        self.caption_drop_rates = caption_drop_rates

    def __len__(self):
        return len(self.ocr_data)

    def __getitem__(self, index):
        data = {}
        assert self.ocr_file is not None
        sample = self.ocr_data[index]
        image_id = sample["image_id"]
        ocr_tokens = sample["ocr_tokens"]
        ocr_info = sample["ocr_info"]
        chosen = image_id + ".jpg"
        
        # original image filename
        filename = self.root_dir/chosen
        if not self.no_hint:
            # hint image filename
            hint_filename = self.hint_folder/chosen
            if not os.path.isfile(hint_filename):
                print("Hint file {} does not exist".format(hint_filename))
                return self.__getitem__(np.random.choice(self.__len__()))
        else:
            hint_filename = None

        for d in self.captions:
            if d["image_id"] == image_id:
                image_captions = d["reference_strs"]
                image_classes = d["image_classes"]
                break

        if not len(ocr_tokens) or not len(image_captions) or not len(image_classes):
            return self.__getitem__(np.random.choice(self.__len__()))
        
        # filter data according the object classes
        if self.filter_data:
            if not len([word for word in self.filter_words if word in " ".join(image_classes).lower()]):
                return self.__getitem__(np.random.choice(self.__len__()))
        

        # get the info about the ocr area in order to 
        # (1): obtain the locations where the images are cropped during augmentation 
        # (2): filter the data according to the number of ocr tokens [No need for the dataset which has been filtered]  
        with Image.open(filename) as img:
            im_w, im_h = img.size 
        pos_info_list = []
        pos_info_tuples = []
        # filter the data according to the number of ocr tokens    
        if self.filter_token_num and len(ocr_info) > self.max_token_num:
            return self.__getitem__(np.random.choice(self.__len__()))
        for item in ocr_info:
            token_box = item['bounding_box']
            lf, up = token_box['top_left_x'], token_box['top_left_y']
            w, h = token_box['width'], token_box['height']
            if not self.new_ocr_info:
                # old version
                rg, dn = lf + w, up + h
                pos_info_list.append([lf, up, rg, dn])
            else:
                ## fix the bug when rotation happens
                lf, w = int(lf * im_w), int(w * im_w)
                up, h = int(up * im_h), int(h * im_h)
                yaw = token_box['yaw']
                tf_xy = np.array([lf, up])
                yaw = yaw * np.pi / 180
                rotate_mx = np.array([
                    [np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw), np.cos(yaw)]
                ])
                rel_cord = np.matmul(rotate_mx, np.array(
                    [[0, 0],
                    [w, 0],
                    [0, h],
                    [w, h]]
                ).T)
                min_xy = np.min(rel_cord, axis = 1).astype(int) + tf_xy
                max_xy = np.max(rel_cord, axis = 1).astype(int) + tf_xy
                pos_info_list.append(
                    [
                    min_xy[0], min_xy[1],
                    max_xy[0], max_xy[1]
                    ]
                )
                mean_xy = rel_cord[:, -1] / 2 + tf_xy
                pos_info_tuples.append((item["word"], 0.2 * lf + mean_xy[1])) #0.15
        pos_info_list = np.array(pos_info_list)
        all_lf, all_up = np.min(pos_info_list[:, :2], axis = 0)
        all_rg, all_dn = np.max(pos_info_list[:, 2:], axis = 0)
        all_pos_info = [all_lf, all_up, all_rg, all_dn]

        # embed the rendered text into the prompt
        caption_wr_text = None
        arrange_tokens = [item[0] for item in (sorted(pos_info_tuples, key=lambda x: x[1]))]
        if self.rendered_txt_in_caption:
            assert self.filter_data # TODO: support other image classes
            valid_words = " ".join(arrange_tokens)
            class_name = ""
            for word in self.filter_words:
                if word in " ".join(image_classes).lower():
                    class_name = word
                    break
            if class_name == "":
                return self.__getitem__(np.random.choice(self.__len__()))
            else:
                caption_wr_text = 'A {} that says "{}".'.format(
                    class_name, valid_words
                    )      
                                 
        # process the original image and hint glyph image
        if self.do_new_proc:
            # recommended
            assert all_pos_info
            im, im_hint = self.new_proc_func(filename, all_pos_info, hint_filename)
        else:
            im_hint = None
            im = Image.open(filename)
            im = self.process_im(im) # Do not support the flip option for now
            if hint_filename is not None:
                im_hint = Image.open(hint_filename)
                im_hint = self.process_im(im_hint) 

        if not self.no_hint:
            assert im_hint is not None
            data[self.control_key] = im_hint
        data[self.first_stage_key] = im

        if self.return_paths:
            data["path"] = str(filename)
       
        caption_ori = random.choice(image_captions)
        caption_wo_text = None # TODO
        prompt_list = []
        for i in range(len(self.caption_choices)):
            cc = self.caption_choices[i]
            if cc == "original":
                caption = caption_ori
            elif cc == "w_rend_text":
                caption = caption_wr_text if caption_wr_text is not None else caption_ori
            elif cc == "wo_rend_text":
                caption = caption_wo_text if caption_wo_text is not None else caption_ori
            
            if torch.rand(1) < self.caption_drop_rates[i]:
                caption = ""
            prompt_list.append(caption)

        data[self.cond_stage_key] = prompt_list if len(prompt_list) > 1 else prompt_list[0]
        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def simple_process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)
    

    


