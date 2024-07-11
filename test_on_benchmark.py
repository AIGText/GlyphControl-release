import argparse
import os
import time
from omegaconf import OmegaConf
from torchvision.transforms import ToTensor
from torch import autocast
from contextlib import nullcontext
from PIL import Image
from scripts.rendertext_tool import Render_Text, load_model_from_config
from glob import glob
from ocr_acc import OCR_EM_Counter,  OCR_EM_without_capitalization_Counter, OCR_Levenshtein_Distance
import easyocr
import numpy as np
import random
from pytorch_lightning import seed_everything
from cldm.hack import disable_verbosity, enable_sliced_attention
import torch


def test(
    args,
    width,
    ratio,
    top_left_x,
    top_left_y,
    yaw,
    num_rows,
    image_resolution,
    strength,
    guess_mode,
    ddim_steps,
    scale,
    eta,
    a_prompt,
    n_prompt,
    seed,
    num_samples,
    save_path,
    render_none,
    only_default_prompt_sd = False,
    default_prompt_sd = "",
):  
    # add seed
    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)
    
    if not args.from_file:
        prompts = [args.prompt]
        data = [args.rendered_txt]
        print("The prompt is {}".format(prompts))
        print("The rendered_txt is {}".format(data))
        assert prompts is not None
        save_path = os.path.join(save_path, args.prompt)
    else:
        print(f"Reading candidate words from {args.from_file}") 
        with open(args.from_file, "r") as f:
            data = f.read().splitlines()
            if args.grams > 1: # default args.grams=1
                data = [" ".join(data[i:i + args.grams]) for i in range(0, len(data), args.grams)]
                
            if args.prompt_from_file: # For CreativeBench, randomly choose different prompt templates
                with open(args.prompt_from_file, "r") as f:
                    prompt_data = [line.strip() for line in f.readlines()]
                    
                def get_random_combinations(prompt_data, data):  
                
                    return [random.choice(prompt_data).replace('""', f'"{sub_data}"') for sub_data in data]
                
                prompts = get_random_combinations(prompt_data, data)
                
            elif "gram" in os.path.basename(args.from_file):

                prompts = ['A sign that says "{}"'.format(line.strip()) for line in data]

           
            if args.max_num_prompts is not None and args.max_num_prompts > 0:
                print("Only use {} prompts.".format(args.max_num_prompts))
                data = data[:args.max_num_prompts]
                prompts = prompts[:args.max_num_prompts]
            
            if not args.prompt_from_file:
                save_path = os.path.join(
                    save_path,
                    os.path.splitext(os.path.basename(args.from_file))[0] 
                    + "_{}_gram".format(args.grams),
                )
            else:
                save_path = os.path.join(
                    save_path,
                    os.path.splitext(os.path.basename(args.from_file))[0] 
                    + f"_prompt_file_{os.path.splitext(os.path.basename(args.prompt_from_file))[0]}",
                )
                
    save_path = os.path.join(
        save_path, 
        os.path.splitext(os.path.basename(args.ckpt))[0]
        ) if not render_none else os.path.join(save_path, "vanilla-stable-diffusion")
    
    if not os.path.exists(save_path):
        print("store generation results to {}".format(save_path))
        os.makedirs(save_path)
    else:
        if args.do_ocr_eval and os.path.exists(os.path.join(save_path, "ocr_results.txt")):
            print("We have store ocr evaluation results at {}".format(save_path))
            if not args.renew_res:
                return
            else:
                print("But we will renew the results")
    print(f"Prompts are {prompts}.")
    if args.deepspeed_ckpt:
        assert os.path.isdir(args.ckpt)
        args.ckpt = os.path.join(args.ckpt, "checkpoint", "mp_rank_00_model_states.pt")
        assert os.path.exists(args.ckpt)


    if args.do_ocr_eval:
        ocr_em_counter = OCR_EM_Counter()
        ocr_em_wc_counter = OCR_EM_without_capitalization_Counter()
        ocr_lev = OCR_Levenshtein_Distance()
        reader = easyocr.Reader(['en'])
    print("The num of samples is {}".format(num_samples))
    cfg = OmegaConf.load(f"{args.cfg}")
    print("Begin load model.")
    start_time = time.time()
    model = load_model_from_config(cfg, f"{args.ckpt}", verbose=True, not_use_ckpt=args.not_use_ckpt)
    print(f"Model has been loaded, which takes {time.time() - start_time}s.")
    precision_scope = autocast if args.precision == "autocast" else nullcontext
    transform = ToTensor()
    
    render_tool = Render_Text(
        model, precision_scope,
        transform, save_memory = args.save_memory,
        )

    for i in range(len(data)):
        inputs = (
            data[i] if not render_none else "", 
            prompts[i] if not only_default_prompt_sd else default_prompt_sd, 
            width, 
            ratio, 
            top_left_x, 
            top_left_y, 
            yaw, 
            num_rows,
            a_prompt, 
            n_prompt, 
            num_samples, 
            image_resolution, 
            ddim_steps, 
            guess_mode, 
            strength, 
            scale, 
            seed, 
            eta
        )
        all_results = render_tool.process(*inputs)
        img_array_list = all_results[1:] if not render_none else all_results
        rendered_text = data[i]
        if args.do_ocr_eval:
            ocr_em_counter.add_text(rendered_text)
            ocr_em_wc_counter.add_text(rendered_text)
            ocr_lev.add_text(rendered_text)
        for idx, result in enumerate(img_array_list):
            if args.do_ocr_eval and idx  < args.ocr_num_per_txt:
                ocr_result = reader.readtext(result)
                ocr_em_counter.update(rendered_text, ocr_result)
                ocr_em_wc_counter.update(rendered_text, ocr_result)
                ocr_lev.update(rendered_text, ocr_result)
            if not args.not_save_img:
                result_im = Image.fromarray(result)
                result_path = os.path.join(save_path, rendered_text)
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                result_im.save(os.path.join(result_path, f"{rendered_text}_{idx}.jpg"))
        if not args.not_save_img:
            prompt_i = prompts[i] if not only_default_prompt_sd else default_prompt_sd
            if a_prompt == "": 
                prompt_i = prompt_i + '.'
            else:
                prompt_i = prompt_i +  ', ' + a_prompt
            with open(os.path.join(result_path, "prompt.txt"), 'w') as fp:
                fp.write( prompt_i + '\n')
        if not render_none and args.save_glyph_images:
            all_results[0].save(os.path.join(result_path, f"{rendered_text}_glyph_image.jpg"))
        torch.cuda.empty_cache()
        
    if args.do_ocr_eval:
        print("----------------------")
        print_str = "OCR results using the {} ckpt on {}\n".format(
            args.ckpt, 
            args.from_file if args.from_file is not None else args.prompt
            )
        print_str += "OCR evaluation on {} images with {} words rendered\n".format(
            # len(ocr_em_counter.ocr_acc_em), 
            np.sum([meter.count for meter in ocr_em_counter.ocr_acc_em.values()]),
            len(ocr_em_counter.ocr_acc_em), 
        )
        print_str += ocr_em_counter.__repr__() + "\n"
        print_str += ocr_em_wc_counter.__repr__() + "\n"
        print_str += ocr_lev.__repr__()
        print(print_str)
        with open(
            os.path.join(save_path, "ocr_results.txt"), "w"
        ) as f:
            f.write(print_str)

    torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/config.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--not_use_ckpt",
        action="store_true",
        help="not to use the ckpt",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="full" #"autocast"
    )
    # prompt settings in the test benchmark
    parser.add_argument(
        "--max_num_prompts",
        type=int,
        default=None,
        help="max num of the used prompts",
    )
    parser.add_argument(
        "--grams",
        type=int,
        default=1,
        help="How many grams (words or symbols) to form the to-be-rendered text (used for DrawSpelling Benchmark)",
    )
    # please use the files in the text_prompts/raw folder
    parser.add_argument(
        "--from-file",
        type=str, 
        default="text_prompts/raw/SimpleBench/all_unigram_1000_10000_100.txt",
        help="if specified, load rendered_words from this file, separated by newlines",
    )
    parser.add_argument(
        "--prompt-from-file",
        type=str,
        # "text_prompts/raw/CreativeBench/GlyphDraw_origin_remove_render_words.txt"
        help="(CreativeBench) if specified, load prompt template from this file, separated by newlines",
    )
    # glyph instructions for all rendered text
    parser.add_argument(
        "--width",
        type=float,
        default=0.3,
        help="image text width",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0,
        help="text width / height ratio",
    )
    parser.add_argument(
        "--yaw",
        type=float,
        default=0,
        help="image text yaw",
    )
    parser.add_argument(
        "--top_left_x",
        type=float,
        default=0.5,
        help="text top left x",
    )
    parser.add_argument(
        "--top_left_y",
        type=float,
        default=0.5,
        help="text top left y",
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        default=1,
        help="how many rows to render",
    )
    # other settings
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="how many samples to produce for each given prompt. A.k.a batch size",
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
        help="guidance scale",
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
    parser.add_argument(
        "--a_prompt",
        type=str,
        default='', #'best quality, extremely detailed',
        help="additional prompt"
    )
    parser.add_argument(
        "--n_prompt",
        type=str,
        default='', #'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality',
        help="negative prompt"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="../evaluate/images",
        help="where to save images"
    )
    # test on a single prompt
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a sign that says 'Stable Diffusion'",
        help="the prompt"
    )
    parser.add_argument(
        "--rendered_txt",
        type=str,
        nargs="?",
        default="",
        help="the text to render"
    )
    # while testing on multiple ckpts
    parser.add_argument(
        "--ckpt_folder",
        type=str,
        help="paths to checkpoints of model, if specified, use the checkpoints in the folder",
    )
    # ckpts of different training steps
    parser.add_argument(
        "--step_interval",
        type=int,
        default=2000,
        help="interval between two nearby selected training steps",
    )
    parser.add_argument(
        "--eval_start",
        type=int,
        default=1000,
        help="the initial training step for evaluation",
    )
    parser.add_argument(
        "--sub",
        type=int,
        default=1,
        help="the names of trainstep checkpoints end with '1000' (sub=0) or '999' (sub=1)",
    )
    # ckpts of different epochs
    parser.add_argument(
        "--epoch_eval",
        action="store_true",
        help="whether to eval the checkpoints of each epoch",
    )
    parser.add_argument(
        "--epoch_interval",
        type=int,
        default=1,
        help="interval between two nearby selected epochs",
    )
    parser.add_argument(
        "--epoch_eval_start",
        type=int,
        default=0,
        help="the initial training epoch for evaluation",
    )
    # ocr evaluation
    parser.add_argument(
        "--do_ocr_eval",
        action="store_true",
        help="whether to directly evaluate the ocr results",
    )
    parser.add_argument(
        "--ocr_num_per_txt",
        type=int,
        default=4,
        help="num of selected images per word while evaluating the ocr accuracy",
    )
    # others
    parser.add_argument(
        "--render_none",
        action="store_true",
        help="not to render text, use the origin stable diffusion",
    )
    parser.add_argument(
        "--only_default_prompt_sd",
        action="store_true",
        help="whether to only use the default a_prompt & n_prompt for stable diffusion branch (i.e., not input the custom prompts)",
    )
    parser.add_argument(
        "--default_prompt_sd",
        type=str,
        default="",
        help="default prompt for the Stable Diffusion branch if only_default_prompt_sd == True"
    )
    parser.add_argument(
        "--not_save_img",
        action="store_true",
        help="whether to save the generated images (default: True)",
    )
    parser.add_argument(
        "--renew_res",
        action="store_true",
        help="whether to renew the existing ocr results",
    )
    parser.add_argument(
        "--deepspeed_ckpt",
        action="store_true",
        help="whether to use deepspeed while training",
    )
    parser.add_argument(
        "--save_memory",
        action= "store_true",
        help="whether to save memory by transferring some unused parts of models to the cpu device during inference",
    ) 
    parser.add_argument(
        "--save_glyph_images",
        action= "store_true",
        help="whether to save glyph images",
    ) 
    return parser

if __name__ == "__main__":
    import sys
    cur_folder = os.path.dirname(os.path.realpath(__file__)) 
    if os.getcwd() != cur_folder:
        os.chdir(cur_folder)
        print(os.getcwd())
    sys.path.append(os.getcwd())
    parser = parse_args()
    args = parser.parse_args()

    disable_verbosity()
    if args.save_memory:
        # save GPU memory usage
        enable_sliced_attention()
    width = args.width
    ratio = args.ratio
    top_left_x = args.top_left_x
    top_left_y = args.top_left_y
    yaw = args.yaw
    num_rows = args.num_rows
    image_resolution = args.image_resolution
    strength  = args.strength
    guess_mode = args.guess_mode
    ddim_steps = args.ddim_steps
    scale = args.scale
    eta = args.eta
    a_prompt = args.a_prompt
    n_prompt = args.n_prompt
    seed = args.seed
    num_samples = args.num_samples
    save_path = args.save_path
    render_none = True if args.render_none else False
    only_default_prompt_sd = args.only_default_prompt_sd
    default_prompt_sd = args.default_prompt_sd
    ckpt_list = [] #specify the checkpoints to be evaluated

    steps = range(args.eval_start - args.sub, 150000 - args.sub, args.step_interval)
    steps = ["step=%09d.ckpt" % step  for step in steps]

    epochs = range(args.epoch_eval_start, 300, args.epoch_interval)
    epochs = ["epoch=%06d.ckpt" % epoch for epoch in epochs]

    if args.ckpt_folder is not None: # multiple checkpoint evaluation
        for ckpt in glob(args.ckpt_folder + "/*.ckpt"):
            if len(ckpt_list): 
                if os.path.basename(ckpt) not in ckpt_list:
                    continue
            else:
                if "last" in os.path.basename(ckpt):
                    continue
                # (modify the codes based on the filenames of your own checkpoints)
                if "trainstep_checkpoints" in args.ckpt_folder and os.path.basename(ckpt).split("-")[1] not in steps:
                    continue   # evaluated on different training steps 
                if args.epoch_eval and os.path.basename(ckpt) not in epochs:
                    continue   # evaluated on different epochs
            print("Test on ", ckpt)
            args.ckpt = ckpt
            test(
                args,
                width,
                ratio,
                top_left_x,
                top_left_y,
                yaw,
                num_rows,
                image_resolution,
                strength,
                guess_mode,
                ddim_steps,
                scale,
                eta,
                a_prompt,
                n_prompt,
                seed,
                num_samples,
                save_path,
                render_none,
                only_default_prompt_sd,
                default_prompt_sd,
            )
    else:
        test(
                args,
                width,
                ratio,
                top_left_x,
                top_left_y,
                yaw,
                num_rows,
                image_resolution,
                strength,
                guess_mode,
                ddim_steps,
                scale,
                eta,
                a_prompt,
                n_prompt,
                seed,
                num_samples,
                save_path,
                render_none,
                only_default_prompt_sd,
                default_prompt_sd,
            )    