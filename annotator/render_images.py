from PIL import Image, ImageFont, ImageDraw
import random
import numpy as np

# resize height to image_height first, then shrink or pad to image_width
def resize_and_pad_image(pil_image, image_size):

    if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        image_width, image_height = image_size
    elif isinstance(image_size, int):
        image_width  = image_height = image_size
    else:
        raise ValueError(f"Image size should be int or list/tuple of int not {image_size}")

    while pil_image.size[1] >= 2 * image_height:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    
    scale = image_height / pil_image.size[1]
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size),resample=Image.BICUBIC)

    # shrink
    if pil_image.size[0] > image_width:
        pil_image = pil_image.resize((image_width, image_height),resample=Image.BICUBIC)
    
    # padding
    if pil_image.size[0] < image_width:
        img = Image.new(mode="RGB",size=(image_width,image_height), color="white")
        width, _ = pil_image.size
        img.paste(pil_image,((image_width - width)//2, 0))
        pil_image = img

    return pil_image

def render_text_image_custom(image_size, bboxes, rendered_txt_values, num_rows_values, font_name="calibri", align = "center"):
    # aligns = ["center", "left", "right"]
    '''
    Render text image based on the glyph instructions, i.e., the list of tuples (text, bbox, num_rows).
        Currently we just use Calibri font to render glyph images.
    '''
    print(image_size, bboxes, rendered_txt_values, num_rows_values, align)
    background = Image.new("RGB", image_size, "white")
    font = ImageFont.truetype(f"fonts/{font_name}.ttf", encoding='utf-8', size=512)
    
    for text, bbox, num_rows in zip(rendered_txt_values, bboxes, num_rows_values):
        
        if len(text) == 0:
            continue
        
        text = text.strip()
        if num_rows != 1:
            word_tokens = text.split()
            num_tokens = len(word_tokens)
            index_list = range(1, num_tokens + 1)
            if num_tokens > num_rows:
                index_list = random.sample(index_list, num_rows)
                index_list.sort()
            line_list = []
            start_idx = 0
            for index in index_list: 
                line_list.append(
                    " ".join(word_tokens
                    [start_idx: index]
                    )
                )
                start_idx = index
            text = "\n".join(line_list)
        
        if 'ratio' not in bbox or bbox['ratio'] == 0 or bbox['ratio'] < 1e-4:
            image4ratio = Image.new("RGB", (512, 512), "white")
            draw = ImageDraw.Draw(image4ratio)
            _, _ , w, h = draw.textbbox(xy=(0,0),text = text, font=font)
            ratio = w / h
        else:
            ratio = bbox['ratio']
        
        width = int(bbox['width'] * image_size[1])
        height = int(width / ratio)
        top_left_x = int(bbox['top_left_x'] * image_size[0])
        top_left_y = int(bbox['top_left_y'] * image_size[1])
        yaw = bbox['yaw']
        
        text_image = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(text_image)
        x,y,w,h = draw.textbbox(xy=(0,0),text = text, font=font)   
        text_image = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(text_image)
        draw.text((-x/2,-y/2), text, "black", font=font, align=align)
        text_image = resize_and_pad_image(text_image, (width, height))
        text_image = text_image.rotate(angle=-yaw, expand=True, fillcolor="white")
        # image = Image.new("RGB", (w, h), "white")
        # draw = ImageDraw.Draw(image)
        
        background.paste(text_image, (top_left_x, top_left_y))
    
    return background

def render_text_image_laionglyph(image_size, ocrinfo, confidence_threshold=0.5):
    '''
    Render the glyph image according to the ocr information for the samples in the LAIONGlyph Dataset 
    '''
    font = ImageFont.truetype("calibri.ttf", encoding='utf-8', size=512)
    background = Image.new("RGB", image_size, "white")
    
    for sub_ocr_info in ocrinfo:
        
        bbox, text, confidence = sub_ocr_info
        
        if confidence < confidence_threshold:
            continue
        
        # print(bbox, text, confidence)
        # Calculate the real size
        real_width = int(bbox[1][0] - bbox[0][0])
        real_height = int(bbox[3][1] - bbox[0][1])
        # Calculate the rotation parameter
        bbox_center = [(bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2]
        angle = np.arctan2(bbox[1][1] - bbox_center[1], bbox[1][0] - bbox_center[0]) * 180 / np.pi
        
        text_image = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(text_image)
        x,y,w,h = draw.textbbox(xy=(0,0),text = text, font=font)   
        text_image = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(text_image)
        draw.text((-x/2,-y/2), text, "black", font=font, align="center")
        
        text_image = resize_and_pad_image(text_image, (real_width, real_height))
        text_image = text_image.rotate(angle=-angle, expand=True, fillcolor="white")
        background.paste(text_image, (int(bbox[0][0]), int(bbox[0][1])))
    
    return background