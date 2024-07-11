import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel, T5ForConditionalGeneration, AutoTokenizer, ByT5Tokenizer
from transformers import AutoProcessor, CLIPVisionModel
import open_clip
from ldm.util import default, count_params, islistortuple
from transformers import PreTrainedTokenizerBase
from ldm.modules.diffusionmodules.util import zero_module, identity_init_fc
class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class', ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0. and not disable_dropout:
            mask = 1. - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1-mask) * torch.ones_like(c)*(self.n_classes-1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class FrozenT5Embedder_old(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""
    def __init__(self, version="google/t5-v1_1-large", device="cuda", max_length=77, freeze=True):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)

class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5/ByT5 transformer encoder for text"""
    def __init__(self, version="google/t5-v1_1-large", device="cuda", max_length=77, freeze=True, padding="max_length"):  
        # version: others for T5 are google/t5-v1_1-xl, google/t5-v1_1-xxl, google/t5-v1_1-small, google/t5-v1_1-base and google/t5-v1_1-large
        #          for ByT5 are google/byt5-small, google/byt5-base, google/byt5-large, google/byt5-xl and google/byt5-xxl
        # padding: "max_length" or "longest" 
        # https://huggingface.co/docs/transformers/v4.24.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version) if "byt5" not in version else ByT5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        self.padding = padding
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding=self.padding, return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        print("Start initializing the CLIP text encoder")
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        print("Initialization ends")
        del model.visual
        self.model = model

        if not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device
            
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        # did not do:  
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection
        # x = F.normalize(x, dim=-1) if normalize else x
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)

class FrozenOpenCLIPSepEncoder(FrozenOpenCLIPEmbedder):
    def forward(self, text):
        if islistortuple(text) and len(text) > 0 and islistortuple(text[0]):
            z_list = []
            for ti in text:
                tokens = open_clip.tokenize(ti)
                z = self.encode_with_transformer(tokens.to(self.device))
                z_list.append(z)
            return z_list
        else:
            tokens = open_clip.tokenize(text)
            z = self.encode_with_transformer(tokens.to(self.device))
            return z


class FrozenCLIPT5Encoder(AbstractEncoder):
    def __init__(self, 
                 clip_version="openai/clip-vit-large-patch14", clip_max_length=77, layer="last", layer_idx=None,
                 t5_version="google/t5-v1_1-xl", t5_max_length=77, padding="max_length",
                 freeze=True, device="cuda"):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(
            clip_version, device, max_length=clip_max_length, freeze=freeze, layer=layer, layer_idx=layer_idx
            )
        self.t5_encoder = FrozenT5Embedder(
            t5_version, device, max_length=t5_max_length, freeze=freeze, padding=padding
            )
        print(f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder)*1.e-6:.2f} M parameters, "
              f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder)*1.e-6:.2f} M params.")

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]

class FrozenOpenCLIPT5Encoder(AbstractEncoder):
    def __init__(self, 
                 arch="ViT-H-14", clip_version="laion2b_s32b_b79k", layer="last", clip_max_length=77,
                 t5_version="google/t5-v1_1-small", t5_max_length=77, padding="max_length",
                 device="cuda", freeze=True):
        super().__init__()
        self.clip_encoder = FrozenOpenCLIPEmbedder(
            arch=arch, version=clip_version, device=device, max_length=clip_max_length,
            freeze=freeze, layer=layer
            )
        self.t5_encoder = FrozenT5Embedder(
            t5_version, device, max_length=t5_max_length, freeze=freeze, padding=padding
            )
        print(f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder)*1.e-6:.2f} M parameters, "
              f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder)*1.e-6:.2f} M params.")

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text) #B*77*1024
        t5_z = self.t5_encoder.encode(text) #B*77*Z
        return [clip_z, t5_z]

class FrozenOpenCLIPT5SepEncoder(FrozenOpenCLIPT5Encoder):
    def forward(self, text):
        if islistortuple(text) and len(text) > 0 and islistortuple(text[0]):
            assert len(text) == 2
            print("two separate input prompts")
            clip_z = self.clip_encoder.encode(text[0]) #B*77*1024
            t5_z = self.t5_encoder.encode(text[1]) #B*77*Z
        else:
            clip_z = self.clip_encoder.encode(text) #B*77*1024
            t5_z = self.t5_encoder.encode(text) #B*77*Z
        return [clip_z, t5_z]

class MergeTextEmb(nn.Module):
    def __init__(self, clip_emb_dim, t5_emb_dim, out_emb_dim=None, trainable=True, merge_mode="add", t5_fc_init="zero"):
        super().__init__()
        out_emb_dim = default(out_emb_dim, clip_emb_dim)
        self.clip_fc = identity_init_fc(nn.Linear(clip_emb_dim, out_emb_dim))
        if t5_fc_init == "zero":
            self.t5_fc = zero_module(nn.Linear(t5_emb_dim, out_emb_dim))
        elif t5_fc_init == "identity":
            self.t5_fc = identity_init_fc(nn.Linear(t5_emb_dim, out_emb_dim))
        else:
            "The initialization way {} is not supported.".format(t5_fc_init)
            raise ValueError
        self.trainable = trainable
        self.merge_mode = merge_mode
        
    def forward(self, clip_emb, t5_emb):
        clip_out = self.clip_fc(clip_emb)
        t5_out = self.t5_fc(t5_emb)
        if self.merge_mode == "concat":
            merge_out = torch.cat([clip_out, t5_out], dim=1)
        elif self.merge_mode == "add":
            assert clip_out.shape == t5_out.shape
            merge_out =  clip_out + t5_out 
        else:
            print("invalid merging way: {}".format(self.merge_mode))
            raise ValueError  
        return merge_out


class TransTextEmb(nn.Module):
    def __init__(self, unet_context_dim, emb_dims, fc_inits=None, trans_trainable = None):
        super().__init__()
        # assert isinstance(emb_dims, list)
        emb_num = len(emb_dims)
        if fc_inits is not None:
            # assert isinstance(fc_inits, list) and 
            assert len(fc_inits) == emb_num
        else:
            fc_inits = ["random" for i in range(emb_num)]

        if trans_trainable is not None:
            # assert isinstance(trans_trainable, list) and 
            assert len(trans_trainable) == emb_num
        else:
            trans_trainable = [True for i in range(emb_num)]

        module_list = nn.ModuleList([])
        for i in range(emb_num):
            trans = nn.Linear(emb_dims[i], unet_context_dim)
            if fc_inits[i] == "zero":
                trans = zero_module(trans)
            elif fc_inits[i] == "identity":
                trans = identity_init_fc(trans)
            module_list.append(trans)
        
        self.trans_list = module_list
        self.trans_trainable = trans_trainable
        self.emb_num = emb_num
        
    def forward(self, emb_list):
        assert len(emb_list) == self.emb_num
        emb_out_list = []
        for i in range(self.emb_num):
            emb_out = self.trans_list[i](emb_list[i])
            emb_out_list.append(emb_out)
        return emb_out_list
    

class FrozenOpenCLIPT5ByT5Encoder(AbstractEncoder):
    def __init__(self, 
                 arch="ViT-H-14", clip_version="laion2b_s32b_b79k", layer="last", clip_max_length=77,
                 t5_version="google/t5-v1_1-large", t5_max_length=77, padding="max_length",
                 byt5_version="google/byt5-large", byt5_max_length=77, byt5_padding="max_length",
                 device="cuda", freeze=True):
        super().__init__()
        self.clip_encoder = FrozenOpenCLIPEmbedder(
            arch=arch, version=clip_version, device=device, max_length=clip_max_length,
            freeze=freeze, layer=layer
            )
        self.t5_encoder = FrozenT5Embedder(
            t5_version, device, max_length=t5_max_length, freeze=freeze, padding=padding
            )
        self.byt5_encoder = FrozenT5Embedder(
            byt5_version, device, max_length=byt5_max_length, freeze=freeze, padding=byt5_padding
            )
        print(f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder)*1.e-6:.2f} M parameters, "
              f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder)*1.e-6:.2f} M params."
              f"{self.byt5_encoder.__class__.__name__} comes with {count_params(self.byt5_encoder)*1.e-6:.2f} M params.")

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text) #B*77*1024
        t5_z = self.t5_encoder.encode(text) #B*77*Z
        byt5_z = self.byt5_encoder.encode(text)
        return [clip_z, t5_z, byt5_z]


class FrozenOpenCLIPT5ByT5SepEncoder(FrozenOpenCLIPT5ByT5Encoder):
    def forward(self, text):
        if islistortuple(text) and len(text) > 0 and islistortuple(text[0]):
            assert len(text) <= 3
            clip_text = text[0]
            t5_text = text[1] if len(text) > 1 else text[0]
            byt5_text = text[-1]
        else:
            clip_text = text
            t5_text = text
            byt5_text = text
        clip_z = self.clip_encoder.encode(clip_text) #B*77*1024
        t5_z = self.t5_encoder.encode(t5_text) #B*77*Z_1
        byt5_z = self.byt5_encoder.encode(byt5_text) #B*77*Z_2
        del clip_text, t5_text, byt5_text
        return [clip_z, t5_z, byt5_z]


class OpenCLIPImageEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for image
    """
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", 
                 freeze=True, set_grad_checkpointing = True):
        super().__init__()
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        self.image_mean = model.visual.image_mean
        self.image_std = model.visual.image_std
        del model.transformer
        del model.token_embedding
        del model.positional_embedding
        del model.ln_final
        del model.text_projection
        del model.logit_scale
        # only model.visual is left
        # open_clip.model._build_vision_tower()

        self.model = model
        self.device = device
        
        if not freeze and set_grad_checkpointing:
            self.model.visual.set_grad_checkpointing(True)
        self.freeze_model = freeze

    def forward(self, img):
        z = self.model.encode_image(img) # 2.0.2 , normalize=False) 2.7.0
        return z

    def encode(self, img):
        return self(img)