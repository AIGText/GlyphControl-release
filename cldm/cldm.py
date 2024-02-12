import einops
import torch
import torch as th
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.ema import LitEma
from contextlib import contextmanager, nullcontext
from cldm.model import load_state_dict
import numpy as np
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, OneCycleLR
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False,  **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(self, 
                 control_stage_config, 
                 control_key, only_mid_control,   
                 learnable_conscale = False, guess_mode=False,
                 sd_locked = True, sep_lr = False, decoder_lr = 1.0**-4, 
                 sep_cond_txt = True, exchange_cond_txt = False, concat_all_textemb = False,
                 *args, **kwargs
                 ): 
        use_ema = kwargs.pop("use_ema", False)
        ckpt_path = kwargs.pop("ckpt_path", None)
        reset_ema = kwargs.pop("reset_ema", False)
        only_model= kwargs.pop("only_model", False)
        reset_num_ema_updates = kwargs.pop("reset_num_ema_updates", False)
        keep_num_ema_updates = kwargs.pop("keep_num_ema_updates", False)
        ignore_keys = kwargs.pop("ignore_keys", [])

        super().__init__(*args, use_ema=False, **kwargs)

        # Glyph ControlNet
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control

        self.learnable_conscale = learnable_conscale
        conscale_init = [1.0] * 13 if not guess_mode else [(0.825 ** float(12 - i)) for i in range(13)]
        if learnable_conscale:
            # self.control_scales = nn.Parameter(torch.ones(13), requires_grad=True)
            self.control_scales = nn.Parameter(torch.Tensor(conscale_init), requires_grad=True)
        else: 
            self.control_scales = conscale_init #[1.0] * 13 
        
        self.optimizer = torch.optim.AdamW   
        # whether to unlock (fine-tune) the decoder parts of SD U-Net  
        self.sd_locked = sd_locked
        self.sep_lr = sep_lr
        self.decoder_lr = decoder_lr
        
        # specify the input text embedding of two branches (SD branch and Glyph ControlNet branch)
        self.sep_cond_txt = sep_cond_txt
        self.concat_all_textemb = concat_all_textemb
        self.exchange_cond_txt = exchange_cond_txt
        
        # ema 
        self.use_ema = use_ema
        if self.use_ema: 
            self.model_ema = LitEma(self.control_model, init_num_updates= 0)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
            if not self.sd_locked: 
                self.model_diffoutblock_ema = LitEma(self.model.diffusion_model.output_blocks, init_num_updates= 0)
                print(f"Keeping diffoutblock EMAs of {len(list(self.model_diffoutblock_ema.buffers()))}.")
                self.model_diffout_ema = LitEma(self.model.diffusion_model.out, init_num_updates= 0)
                print(f"Keeping diffout EMAs of {len(list(self.model_diffout_ema.buffers()))}.")
        
        # initialize the model from the checkpoint
        if ckpt_path is not None:
            ema_num_updates = self.init_from_ckpt(ckpt_path, ignore_keys, only_model=only_model)
            self.restarted_from_ckpt = True
            if self.use_ema and reset_ema:
                print(
                    f"Resetting ema to pure model weights. This is useful when restoring from an ema-only checkpoint.")
                self.model_ema = LitEma(self.control_model, init_num_updates= ema_num_updates if keep_num_ema_updates else 0)
                if not self.sd_locked: 
                    self.model_diffoutblock_ema = LitEma(self.model.diffusion_model.output_blocks, init_num_updates= ema_num_updates if keep_num_ema_updates else 0)
                    self.model_diffout_ema = LitEma(self.model.diffusion_model.out, init_num_updates= ema_num_updates if keep_num_ema_updates else 0)

        if reset_num_ema_updates:
            print(" +++++++++++ WARNING: RESETTING NUM_EMA UPDATES TO ZERO +++++++++++ ")
            assert self.use_ema
            self.model_ema.reset_num_updates()
            if not self.sd_locked: # Update
                self.model_diffoutblock_ema.reset_num_updates()
                self.model_diffout_ema.reset_num_updates()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema: # TODO: fix the bug while adding transemb_model or trainable control scales
            self.model_ema.store(self.control_model.parameters())
            self.model_ema.copy_to(self.control_model)
            if not self.sd_locked: # Update
                self.model_diffoutblock_ema.store(self.model.diffusion_model.output_blocks.parameters())
                self.model_diffoutblock_ema.copy_to(self.model.diffusion_model.output_blocks)
                self.model_diffout_ema.store(self.model.diffusion_model.out.parameters())
                self.model_diffout_ema.copy_to(self.model.diffusion_model.out)

            if context is not None:
                print(f"{context}: Switched ControlNet to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.control_model.parameters())
                if not self.sd_locked: # Update
                    self.model_diffoutblock_ema.restore(self.model.diffusion_model.output_blocks.parameters())
                    self.model_diffout_ema.restore(self.model.diffusion_model.out.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights of ControlNet")

    @torch.no_grad()
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):

        if path.endswith("model_states.pt"):
            sd = torch.load(path, map_location='cpu')["module"]
        else:
            # sd = load_state_dict(path, location='cpu') # abandoned
            sd = torch.load(path, map_location="cpu")
            if "state_dict" in list(sd.keys()):
                sd = sd["state_dict"]
    
        keys_ = list(sd.keys())[:]
        for k in keys_:
            if k.startswith("module."):
                nk = k[7:]
                sd[nk] = sd[k]
                del sd[k]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)

        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys:\n {missing}")
        if len(unexpected) > 0:
            print(f"\nUnexpected Keys:\n {unexpected}")
        
        if "model_ema.num_updates" in sd and "model_ema.num_updates" not in unexpected:
            return sd["model_ema.num_updates"].item()
        else: 
            return 0

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=[c] if not isinstance(c, list) else c, c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt_list = cond["c_crossattn"]

        assert len(cond_txt_list) > 0
        # cond_txt: input text embedding of the pretrained SD branch
        # cond_txt_2: input text embedding of the Glyph ControlNet branch
        cond_txt = cond_txt_list[0]
        if len(cond_txt_list) == 1:
            cond_txt_2 = None
        else:
            if self.sep_cond_txt:
                # use each embedding for each branch separately
                cond_txt_2 = cond_txt_list[1]
            else:
                # concat the embedding for Glyph ControlNet branch
                if not self.concat_all_textemb:
                    cond_txt_2 = torch.cat(cond_txt_list[1:], 1) 
                else:
                    cond_txt_2 = torch.cat(cond_txt_list, 1)

        if self.exchange_cond_txt:
            # exchange the input text embedding of two branches
            txt_buffer = cond_txt
            cond_txt = cond_txt_2
            cond_txt_2 = txt_buffer 

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            hint = torch.cat(cond['c_concat'], 1)
            control = self.control_model(x=x_noisy, hint=hint, timesteps=t, context=cond_txt if cond_txt_2 is None else cond_txt_2)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps
    
    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss = super().training_step(batch, batch_idx, optimizer_idx)
        if self.use_scheduler and not self.sd_locked and self.sep_lr:
            decoder_lr = self.optimizers().param_groups[1]["lr"]
            self.log('decoder_lr_abs', decoder_lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())        
        if self.learnable_conscale:
            params += [self.control_scales]
        
        params_wlr = []
        decoder_params = None
        if not self.sd_locked:
            decoder_params = list(self.model.diffusion_model.output_blocks.parameters())
            decoder_params += list(self.model.diffusion_model.out.parameters())
            if not self.sep_lr:
                params.extend(decoder_params)
                decoder_params = None
                
        params_wlr.append({"params": params, "lr": lr})
        if decoder_params is not None:
            params_wlr.append({"params": decoder_params, "lr": self.decoder_lr})
        
        # opt = torch.optim.AdamW(params_wlr) 
        opt = self.optimizer(params_wlr)
        opts = [opt]

        # updated
        schedulers = []
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler_func = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            schedulers = [
                {
                    'scheduler': LambdaLR(
                opt, 
                lr_lambda= [scheduler_func.schedule] * len(params_wlr) #if not self.sep_lr else [scheduler_func.schedule, scheduler_func.schedule]
                ),
                    'interval': 'step',
                    'frequency': 1
                }]
            
        return opts, schedulers

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

    # ema
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.control_model)
            if not self.sd_locked: # Update
                self.model_diffoutblock_ema(self.model.diffusion_model.output_blocks)
                self.model_diffout_ema(self.model.diffusion_model.out)
        if self.log_all_grad_norm:
            zeroconvs = list(self.control_model.input_hint_block.named_parameters())[-2:]
            zeroconvs.extend(
                list(self.control_model.zero_convs.named_parameters())                
            )
            for item in zeroconvs:
                self.log(
                    "zero_convs/{}_norm".format(item[0]),
                    item[1].cpu().detach().norm().item(),
                    prog_bar=False, logger=True, on_step=True, on_epoch=False
                    )
                self.log(
                    "zero_convs/{}_max".format(item[0]),
                    torch.max(item[1].cpu().detach()).item(), #TODO: lack torch.abs
                    prog_bar=False, logger=True, on_step=True, on_epoch=False
                )
            gradnorm_list = []
            for param_group in self.trainer.optimizers[0].param_groups:
                for p in param_group['params']:
                    # assert p.requires_grad and p.grad is not None
                    if p.requires_grad and p.grad is not None:
                        grad_norm_v = p.grad.cpu().detach().norm().item()
                        gradnorm_list.append(grad_norm_v)
            if len(gradnorm_list):
                self.log("all_gradients/grad_norm_mean", 
                    np.mean(gradnorm_list), 
                    prog_bar=False, logger=True, on_step=True, on_epoch=False
                )
                self.log("all_gradients/grad_norm_max", 
                    np.max(gradnorm_list), 
                    prog_bar=False, logger=True, on_step=True, on_epoch=False
                )
                self.log("all_gradients/grad_norm_min", 
                    np.min(gradnorm_list), 
                    prog_bar=False, logger=True, on_step=True, on_epoch=False
                ) 
                self.log("all_gradients/param_num", 
                    len(gradnorm_list), 
                    prog_bar=False, logger=True, on_step=True, on_epoch=False
                )

            if self.learnable_conscale:
                for i in range(len(self.control_scales)):
                    self.log(
                        "control_scale/control_{}".format(i),
                        self.control_scales[i],
                        prog_bar=False, logger=True, on_step=True, on_epoch=False
                    )
            del gradnorm_list
            del zeroconvs
