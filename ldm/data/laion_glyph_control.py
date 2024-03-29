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
from ldm.data.util import new_process_im_base #, imagenet_process_im
from glob import glob
import random
import base64
from io import BytesIO
from annotator.render_images import render_text_image_laionglyph


class LAIONGlyphCLDataset(Dataset):
    
    '''
    data class for LAIONGlyph dataset

    Input: 
        data_folder: the folder storing the data json files.
        data_info_file: the tsv file recording the location of each sample
            The file for 10M dataset should look like this:
                LAION-Glyph-10M_0.json\t0
                LAION-Glyph-10M_0.json\t1
                ...
                LAION-Glyph-10M_1.json\t0
                LAION-Glyph-10M_1.json\t1
                ...
        
    '''
    def __init__(self,

        data_folder,
        data_info_file, 

        max_num_samples = -1, 
        no_hint = False, 

        first_stage_key = "jpg", 
        cond_stage_key = "txt",
        control_key = "hint",
        BLIP_caption = False, #True,
        ocr_threshold = 0.5,

        rendered_txt_in_caption = False,
        caption_choices = ["original", "w_rend_text", "wo_rend_text"],
        caption_drop_rates = [0.1, 0.5, 0.1],

        postprocess=None,
        new_proc_config = None,
        rm_text_from_cp = False,
        replace_token = "",
        ) -> None:
        with open(data_info_file, "r") as f:
            data_infos = f.readlines()
        if max_num_samples > 0:
            data_infos = random.sample(data_infos, max_num_samples)
        self.data_infos = data_infos
        self.data_folder = data_folder

        self.ocr_threshold = ocr_threshold # the threshold of OCR recognition confidence 
        self.no_hint = no_hint
        
        self.caption_choices = caption_choices
        self.caption_drop_rates = caption_drop_rates # random drop caption
        self.rendered_txt_in_caption = rendered_txt_in_caption
        self.BLIP_caption = BLIP_caption # whether to use the captions generated by BLIP-2
        
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.control_key = control_key

        # postprocess
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess

        # image transform
        if new_proc_config is not None:
            self.new_proc_func = instantiate_from_config(new_proc_config)
        else:
            self.new_proc_func = new_process_im_base()
        
        self.rm_text_from_cp = rm_text_from_cp
        self.replace_token = replace_token


    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        data = {}
        # data info
        data_info = self.data_infos[index]
        filename, idx_in_file = data_info.split("\t")[:]
        idx_in_file = int(idx_in_file.strip())  
        with open(os.path.join(self.data_folder, filename), "r") as f:
            ori_data = json.load(f)[idx_in_file]
        img_id = ori_data["img_id"]

        # 1. Load the original image
        img_code = ori_data["img_code"] 
        try:
            ori_img = Image.open(BytesIO(base64.b64decode(img_code)))
        except:
            print("can't open original image: {}".format(img_id))
            return self.__getitem__(np.random.choice(self.__len__())) 

        # 2. Load the caption
        if self.BLIP_caption:
            caption_ori = ori_data["caption_blip"]
        else:
            caption_ori = ori_data["caption_origin"]
        img_size = ori_img.size
        
        # 3. Load ocr info
        ocr_info = data["ocr_info"]
        
        pos_info_list = []
        pos_info_tuples = []
        for info in ocr_info:
            bbox, (text, confidence) = info
            if confidence > self.ocr_threshold:
                xy_info = np.array(bbox)
                min_x, min_y = np.min(xy_info, axis = 0).astype(int)
                max_x, max_y = np.max(xy_info, axis = 0).astype(int)
                pos_info_list.append(
                    [min_x, min_y, max_x, max_y]
                )
                mean_xy = (xy_info[0] + xy_info[2]) / 2
                lf = xy_info[0, 0] # min_x
                pos_info_tuples.append((text, 0.2 * lf + mean_xy[1])) #0.15
                ocr_txt = info[1]
            
        pos_info_list = np.array(pos_info_list)
        all_lf, all_up = np.min(pos_info_list[:, :2], axis = 0)
        all_rg, all_dn = np.max(pos_info_list[:, 2:], axis = 0)
        all_pos_info = [all_lf, all_up, all_rg, all_dn]

        # hint glyph image
        if not self.no_hint:
            try:
                hint_img = render_text_image_laionglyph(
                    img_size, ocr_info, self.ocr_threshold
                )
            except:
                print("can't render hint image: {}".format(img_id))
                return self.__getitem__(np.random.choice(self.__len__()))
        else:
            hint_img = None

        assert all_pos_info
        im, im_hint = self.new_proc_func(ori_img, all_pos_info, hint_img)
        
        if not self.no_hint:
            assert im_hint is not None
            data[self.control_key] = im_hint
        data[self.first_stage_key] = im

        caption_wr_text = None
        arrange_tokens = [item[0] for item in (sorted(pos_info_tuples, key=lambda x: x[1]))]
        if self.rendered_txt_in_caption:
            valid_words = " ".join(arrange_tokens)
            caption_wr_text = caption_ori + '. Words in the image: "{}"'.format(valid_words)
                          
        # process the ori
        caption_wo_text = None # 
        if self.rm_text_from_cp and self.BLIP_caption: 
            # [Default: False] remove the rendered words from the caption  while using BLIP captions
            caption_items = caption_ori.split(" ")
            lower_arrange_tokens = [tk.lower() for tk in arrange_tokens]
            caption_wo_text = []
            for cp_item in caption_items:
                if cp_item.lower() in lower_arrange_tokens:
                    if self.replace_token != "":
                        caption_wo_text.append(self.replace_token) 
                else:
                    caption_wo_text.append(cp_item)
            caption_wo_text = " ".join(caption_wo_text)
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
