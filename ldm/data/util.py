import torch

from ldm.modules.midas.api import load_midas_transform
import albumentations
from torchvision import transforms
from PIL import Image
import numpy as np
from einops import rearrange
import cv2
from ldm.util import instantiate_from_config
from omegaconf import ListConfig
from open_clip.transform import ResizeMaxSize
class AddMiDaS(object):
    def __init__(self, model_type):
        super().__init__()
        self.transform = load_midas_transform(model_type)

    def pt2np(self, x):
        x = ((x + 1.0) * .5).detach().cpu().numpy()
        return x

    def np2pt(self, x):
        x = torch.from_numpy(x) * 2 - 1.
        return x

    def __call__(self, sample):
        # sample['jpg'] is tensor hwc in [-1, 1] at this point
        x = self.pt2np(sample['jpg'])
        x = self.transform({"image": x})["image"]
        sample['midas_in'] = x
        return sample

class new_process_im_base:
    def __init__(self, 
                 size = 512,
                 interpolation = 3, 
                 do_flip = True,
                 flip_p = 0.5,
                 hint_range_m11 = False,
                 ):
        self.do_flip = do_flip
        self.flip_p = flip_p
        self.rescale = transforms.Resize(size=size, interpolation=interpolation)
        if self.do_flip:
            self.flip = transforms.functional.hflip
        # base_tf [-1, 1]
        base_tf_m11 = [ transforms.ToTensor(), # to be checked
                        transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))]
        self.base_tf_m11 = transforms.Compose(base_tf_m11)
        # base_tf [0, 1]
        base_tf_01 = [ transforms.ToTensor(), # to be checked
                        transforms.Lambda(lambda x: rearrange(x, 'c h w -> h w c'))]
        self.base_tf_01 = transforms.Compose(base_tf_01)
        self.hint_range_m11 = hint_range_m11

    def __call__(self, im, pos_info, im_hint = None):
        # im = Image.open(filename)
        im = im.convert("RGB")
        # crop
        size = im.size
        crop_size = min(size)
        crop_axis = size.index(crop_size)
        lf, up, rg, dn = pos_info
        if crop_axis == 0:
            # width
            box_up, box_dn = self.generate_range(up, dn, size[1], size[0])
            box_lf, box_rg = 0, size[0]
        else:
            box_lf, box_rg = self.generate_range(lf, rg, size[0], size[1])
            box_up, box_dn = 0, size[1]
        im = im.crop((box_lf, box_up, box_rg, box_dn))
        # rescale
        im = self.rescale(im)
        # 
        flip_img = False
        if self.do_flip:
            if torch.rand(1) < self.flip_p:
                im = self.flip(im)
                flip_img = True
        im = self.base_tf_m11(im)
        # im_hint = None
        # if hint_filename is not None:
            # im_hint = Image.open(hint_filename)
        if im_hint is not None:
            im_hint = im_hint.convert("RGB")
            im_hint = im_hint.crop((box_lf, box_up, box_rg, box_dn))
            im_hint = self.rescale(im_hint)
            if flip_img:
                im_hint = self.flip(im_hint)
            im_hint = self.base_tf_m11(im_hint) if self.hint_range_m11 else self.base_tf_01(im_hint)
        return im, im_hint

    def generate_range(self, low, high, len_max, len_min):
        mid = (low + high) / 2 * (len_max if high <= 1 else 1) 
        max_range = min(mid + len_min / 2, len_max)
        min_range = min(
            max(mid - len_min / 2, 0 ), 
            max(max_range - len_min, 0)
            )
        return int(min_range), int(min_range + len_min)

class new_process_im(new_process_im_base):
    def __call__(self, filename, pos_info, hint_filename = None):
        im = Image.open(filename) 
        if hint_filename is not None:
            im_hint = Image.open(hint_filename)
        else:
            im_hint = None
        return super().__call__(im, pos_info, im_hint)
        
class imagenet_process_im:
    def __init__(self, 
        size = 512,
        do_flip = False,
        min_crop_f=0.5, 
        max_crop_f=1., 
        flip_p=0.5,
        random_crop=False
            ):

        self.do_flip = do_flip
        if self.do_flip:
            self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        # self.base = self.get_base()
    
        # self.size = size
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert(max_crop_f <= 1.)
        self.center_crop = not random_crop
        self.image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)
        self.size = size

    def __call__(self, im):
        im = im.convert("RGB")
        image = np.array(im).astype(np.uint8)
        # if image.shape[0] < self.size or image.shape[1] < self.size:
        #     return None
        # crop
        min_side_len = min(image.shape[:2])
        crop_side_len = min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        crop_side_len = int(crop_side_len)
        if self.center_crop:
            self.cropper = albumentations.CenterCrop(height=crop_side_len, width=crop_side_len)
        else:
            self.cropper = albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)
        image = self.cropper(image=image)["image"] # ?
        # rescale
        image = self.image_rescaler(image=image)["image"]
        # flip
        if self.do_flip:
            image = self.flip(Image.fromarray(image))
            image = np.array(image).astype(np.uint8)
        return (image/127.5 - 1.0).astype(np.float32)
    
# used for CLIP image encoder
class process_wb_im:
    def __init__(self, 
        size = 224,
        # do_padding = True,
        image_transforms=[], 
        use_clip_resize=False,
        image_mean = None,
        image_std = None,
        exchange_channel = True, 
        ):
        self.image_rescaler = albumentations.LongestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)
        self.image_size = size
        # self.do_padding = do_padding
        self.pad = albumentations.PadIfNeeded(min_height= self.image_size, min_width=self.image_size,
                                              border_mode=cv2.BORDER_CONSTANT, value= (255, 255, 255), 
                                              )
        if isinstance(image_transforms, ListConfig):
            image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean= image_mean if image_mean is not None else (0.48145466, 0.4578275, 0.40821073), 
                std= image_std if image_std is not None else (0.26862954, 0.26130258, 0.27577711)
                ),
                ])
            # transforms.Lambda(lambda x: rearrange(x, 'c h w -> h w c'))
            # ]) #  transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        if exchange_channel:
            image_transforms.append(
                transforms.Lambda(lambda x: rearrange(x, 'c h w -> h w c'))
            )
        image_transforms = transforms.Compose(image_transforms)          
        self.tform = image_transforms
        self.use_clip_resize = use_clip_resize
        self.clip_resize = ResizeMaxSize(max_size = self.image_size, interpolation=transforms.InterpolationMode.BICUBIC, fill=(255, 255, 255))
        
    def __call__(self, im):
        im = im.convert("RGB")
        # if self.do_padding:
        #     im = self.padding_image(im)
        if self.use_clip_resize:
            im = self.clip_resize(im)
        else:
            im = self.padding_image(im)
        return self.tform(im)

    
    def padding_image(self, im):
        # resize 
        im = np.array(im).astype(np.uint8)
        im_rescaled = self.image_rescaler(image=im)["image"]
        # padding
        im_padded = self.pad(image=im_rescaled)["image"]
        return im_padded

# use for VQ-GAN
class vqgan_process_im:
    def __init__(self, size=384, random_crop=False, augment=False, ori_preprocessor = False, to_tensor=False):
        self.size = size
        self.random_crop = random_crop
        self.augment = augment
        assert self.size is not None and self.size > 0
        if ori_preprocessor:
            # if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop: # train
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size) 
            else: # test
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
            # else:
            #     self.preprocessor = lambda **kwargs: kwargs
        else:
            self.rescaler = albumentations.LongestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)
            self.pad = albumentations.PadIfNeeded(min_height= self.size, min_width=self.size,
                                              border_mode=cv2.BORDER_CONSTANT, value= (255, 255, 255), 
                                              )
            self.preprocessor = albumentations.Compose([self.rescaler, self.pad])

        if self.augment: # train
            # Add data aug transformations
            self.data_augmentation = albumentations.Compose([
                albumentations.GaussianBlur(p=0.1),
                albumentations.OneOf([
                    albumentations.HueSaturationValue (p=0.3),
                    albumentations.ToGray(p=0.3),
                    albumentations.ChannelShuffle(p=0.3)
                ], p=0.3)
            ])
        
        if to_tensor:
            self.tform = transforms.ToTensor()
        self.to_tensor = to_tensor
        
        # if exchange_channel:
        #     self.exchange_channel = transforms.Lambda(lambda x: rearrange(x, 'c h w -> h w c'))

    def __call__(self, image):
        image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        if self.augment:
            image = self.data_augmentation(image=image)['image']
        image = (image/127.5 - 1.0).astype(np.float32)
        if self.to_tensor:
            image = self.tform(image)
        return image