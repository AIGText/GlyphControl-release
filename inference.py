# ------------------------------------------
# GlyphControl: Glyph Conditional Control for Visual Text Generation
# Paper Link: https://arxiv.org/pdf/2305.18259
# Code Link: https://github.com/AIGText/GlyphControl-release
# This script is used for inference.
# ------------------------------------------


import torch
import time
from PIL import Image
from cldm.hack import disable_verbosity, enable_sliced_attention
from scripts.rendertext_tool import Render_Text, load_model_from_config
from omegaconf import OmegaConf
import argparse
import os
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/config.yaml",
        help="path to model config",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/laion10M_epoch_6_model_ema_only.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="generated_images",
        help="where to save images"
    )
    parser.add_argument(
        "--save_memory",
        type=str,
        default="whether to save memory by transferring some unused parts of models to the cpu device during inference",
        help="path to checkpoint of model",
    )  
    # specify the inference settings
    parser.add_argument(
        "--glyph_instructions",
        type=str,
        default=None, #"glyph_instructions.yaml",
        help="path to glyph instructions",
    )  
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="A sign that says 'APPLE'",
        help="the prompt"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--a_prompt",
        type=str,
        default='4K, dslr, best quality, extremely detailed',
        help="additional prompt"
    )
    parser.add_argument(
        "--n_prompt",
        type=str,
        default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality',
        help="negative prompt"
    )
    parser.add_argument(
        "--image_resolution",
        type=int,
        default=512,
        help="image resolution",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1,
        help="control strength",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="classifier-free guidance scale",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=20,
        help="ddim steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed",
    )
    parser.add_argument(
        "--guess_mode",
        action="store_true",
        help="whether use guess mode",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0,
        help="eta",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    disable_verbosity()
    if args.save_memory:
        enable_sliced_attention()

    try:
        # Glyph Instructions
        glyph_instructions = OmegaConf.load(args.glyph_instructions).Instructions
        # print(glyph_instructions)
        rendered_txt_values = glyph_instructions.rendered_txt_values
        width_values = glyph_instructions.width_values
        ratio_values = glyph_instructions.ratio_values
        top_left_x_values = glyph_instructions.top_left_x_values
        top_left_y_values = glyph_instructions.top_left_y_values
        yaw_values = glyph_instructions.yaw_values
        num_rows_values = glyph_instructions.num_rows_values
        # print(rendered_txt_values, width_values, ratio_values, top_left_x_values, top_left_y_values, yaw_values, num_rows_values)
    except Exception as e:
        print(e)
        rendered_txt_values = [""]
        width_values, ratio_values, top_left_x_values, top_left_y_values, yaw_values, num_rows_values = [None] * 6 

    cfg = OmegaConf.load(args.cfg)
    model = load_model_from_config(cfg, args.ckpt, verbose=True)
    render_tool = Render_Text(model, save_memory = args.save_memory)

    # Render glyph images and generate corresponding visual text 
    # print(args.prompt)
    results = render_tool.process_multi(rendered_txt_values, args.prompt,  
                                     width_values, ratio_values,  
                                     top_left_x_values, top_left_y_values,  
                                     yaw_values, num_rows_values,  
                                     args.num_samples, args.image_resolution,  
                                     args.ddim_steps, args.guess_mode,  
                                     args.strength, args.scale, args.seed,  
                                     args.eta, args.a_prompt, args.n_prompt) 
    
    
    result_path = os.path.join(args.save_path, args.prompt)
    os.makedirs(result_path, exist_ok=True)
    render_none = len([1 for rendered_txt in rendered_txt_values if rendered_txt != ""]) == 0
    if render_none:
        for idx, result in enumerate(results):
            result_im = Image.fromarray(result)
            result_im.save(os.path.join(result_path, f"{idx}.jpg"))
    else:
        rendered_txt_join = "_".join(rendered_txt_values)
        results[0].save(os.path.join(result_path, f"{rendered_txt_join}_glyph_image.jpg"))
        for idx, result in enumerate(results[1:]):
            result_im = Image.fromarray(result)
            result_im.save(os.path.join(result_path, f"{rendered_txt_join}_{idx}.jpg")) 



