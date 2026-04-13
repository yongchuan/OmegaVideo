# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# SPRINT: SPRINT: Sparse-Dense Residual Fusion for Efficient Diffusion Transformers
# --------------------------------------------------------
from einops import rearrange
from typing import Optional, Tuple, Union
import torch
import numpy as np
import math
from timm.models.vision_transformer import Mlp
import torch.nn.functional as F
from collections.abc import Iterable
from itertools import repeat
from torch import Tensor, nn
from models.rope import RoPE3D, PositionGetter3D
from models.builder import build_act, build_norm

def val2list(x: list or tuple or any, repeat_time=1) -> list:  # type: ignore
    """Repeat `val` for `repeat_time` times and return the list or val if list/tuple."""
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]

def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:  # type: ignore
    """Return tuple with min_len by repeating element at idx_repeat."""
    # convert to list first
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)

def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)

def get_same_padding(kernel_size: Union[int, Tuple[int, ...]]) -> Union[int, Tuple[int, ...]]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, f"kernel size {kernel_size} should be odd number"
        return kernel_size // 2

class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, act_layer=nn.GELU(approximate='tanh'), token_num=120):
        super().__init__()
        self.y_proj = Mlp(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size,
                          act_layer=act_layer, drop=0)
        self.register_buffer("y_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0]).to(caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.y_embedding, caption)
        # caption = caption.reshape(caption.shape[0], 77, caption.shape[3])
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        # print(caption.shape)
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        # else:
        # caption = caption.reshape(caption.shape[0], 77, caption.shape[3])
        caption = self.y_proj(caption)

        return caption


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                  These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            norm_layer: nn.Module = nn.RMSNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # for API compatibility with timm Attention
        self.fused_attn = False

        self.rope = RoPE3D()
        self.position_getter = PositionGetter3D()

    def forward(
            self,
            x: torch.Tensor,
            TT: int,
            #thw: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        """Multi-head self-attention with optional 2D RoPE.

        x: (B, N, C)
        rope: VisionRotaryEmbeddingFast or None
        rope_ids: (B, N) or (N,) original (flattened) token indices for RoPE,
                  supporting routed / sparse subsets.
        """
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)

        #if thw is None:
        #    raise ValueError("Video attention requires dynamic (T, H, W) token shape information.")
        #t, h, w = thw
        #pos_thw = self.position_getter(B, t=t, h=h, w=w, device=q.device)
        
        #print("N:", N)
        if N == 2640 + TT:
            pos_thw = self.position_getter(B, t=3, h=22, w=40, device=q.device)
        elif N == 1530 + TT:
            pos_thw = self.position_getter(B, t=3, h=17, w=30, device=q.device)
        elif N == 6120 + TT:
            pos_thw = self.position_getter(B, t=3, h=34, w=60, device=q.device)
        elif N == 660 + TT:
            pos_thw = self.position_getter(B, t=3, h=11, w=20, device=q.device)
        else:
            pos_thw = self.position_getter(B, t=3, h=23, w=40, device=q.device)

        q = self.rope(q, pos_thw, TT)
        k = self.rope(k, pos_thw, TT)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# 激活函数: relu ,sigmoid 》 -grad = 0,sigmoid数值越大=grad掉失越大,tahn，silu
class SwiGLUFFN(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    # @torch.compile
    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

class ConvLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        padding: int or None = None,
        use_bias=False,
        dropout=0.0,
        norm="bn2d",
        act="relu",
    ):
        super().__init__()
        if padding is None:
            padding = get_same_padding(kernel_size)
            padding *= dilation

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.use_bias = use_bias

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_dim)
        self.act = build_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_feature=None,
        kernel_size=3,
        stride=1,
        padding: int or None = None,
        use_bias=False,
        norm=(None, None, None),
        act=("silu", "silu", None),
        dilation=1,
    ):
        out_feature = out_feature or in_features
        super().__init__()
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act = val2tuple(act, 3)

        self.glu_act = build_act(act[1], inplace=False)
        self.inverted_conv = ConvLayer(
            in_features,
            hidden_features * 2,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act=act[0],
        )
        self.depth_conv = ConvLayer(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size,
            stride=stride,
            groups=hidden_features * 2,
            padding=padding,
            use_bias=use_bias[1],
            norm=norm[1],
            act=None,
            dilation=dilation,
        )
        self.point_conv = ConvLayer(
            hidden_features,
            out_feature,
            1,
            use_bias=use_bias[2],
            norm=norm[2],
            act=act[2],
        )
        # from IPython import embed; embed(header='debug dilate conv')

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.point_conv(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)

        return x

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, TT=77, **block_kwargs):
        super().__init__()
        self.TT = TT
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=False,
        )
        if "fused_attn" in block_kwargs.keys():
            self.attn.fused_attn = block_kwargs["fused_attn"]
        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        # mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # approx_gelu = lambda: nn.GELU(approximate="tanh")
        # self.mlp = Mlp(
        #     in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        # )
        # self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        mlp_acts = ("silu", "silu", None)
        self.mlp = GLUMBConv(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            use_bias=(True, True, False),
            norm=(None, None, None),
            act=mlp_acts,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(
            self,
            x: torch.Tensor,
            c: torch.Tensor,
            TT: int,
            T: int,
            H: int,
            W: int,
            S: int
    ) -> torch.Tensor:
        aout = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            aout.chunk(6, dim=1)
        )
        norm_x = self.norm1(x)
        attnInput = modulate(norm_x, shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            attnInput, self.TT
        )

        x_m = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x_m_txt = x_m[:, :TT, :]
        x_m_img = rearrange(x_m[:, TT:, :], "B (T S) C -> (B T) S C", T=T, S=S)
        x_mlp_img = self.mlp(x_m_img, H, W)
        x_mlp_img = rearrange(x_mlp_img, "(B T) S C -> B (T S) C", T=T, S=S)
        x_mlp = torch.cat([x_m_txt, x_mlp_img], dim=1)
        x = x + gate_mlp.unsqueeze(1) * x_mlp
        # x = x + gate_mlp.unsqueeze(1) * self.mlp(
        #     modulate(self.norm2(x), shift_mlp, scale_mlp)
        # )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        # self.adaLN_modulation_act = nn.SiLU()
        # self.adaLN_modulation_linear1 = nn.Linear(hidden_size, hidden_size, bias=True)
        # self.adaLN_modulation_linear2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        # act = self.adaLN_modulation_act(c)
        # shift = self.adaLN_modulation_linear1(act)
        # scale = self.adaLN_modulation_linear2(act)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_height=224,
        img_width=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        kernel_size=None,
        padding=0,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        kernel_size = kernel_size or patch_size
        img_size = (img_height, img_width)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        if not padding and kernel_size % 2 > 0:
            padding = get_same_padding(kernel_size)
        #print("in_chans",in_chans)
        #self.channel_mapper = nn.Conv2d(256, 128, kernel_size=1)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size, padding=padding, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        assert (W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        #x = self.channel_mapper(x)
        #print(x.shape)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone + SPRINT sparse-dense residual fusion.
    """

    def __init__(
            self,
            path_type='edm',
            input_height=11,
            input_width=20,
            patch_size=1,
            in_channels=128,
            hidden_size=1152,
            decoder_hidden_size=1152,
            encoder_depth=8,
            depth=16,
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=1000,
            use_cfg=False,
            projector_dim=2048,
            cls_token_dim=768,
            **block_kwargs  # fused_attn
    ):
        super().__init__()
        self.path_type = path_type
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.gradient_checkpointing = False
        self.use_cfg = use_cfg
        self.num_classes = num_classes
        self.encoder_depth = encoder_depth
        self.depth = depth
        self.contxt_len = 77

        # ----------------- SPRINT configuration -----------------
        # fθ / gθ / hθ split: default 2 / (D-4) / 2 as in the paper.
        self.num_f = 2
        self.num_h = 2
        self.num_g = self.depth - self.num_f - self.num_h
        assert self.num_g >= 0, "depth too small for SPRINT split"

        # Token drop ratio r (fraction of tokens to drop in sparse path)
        self.sprint_drop_ratio = 0.0

        # Path-drop learning probability p (drop whole sparse path during training)
        self.path_drop_prob = 0.05

        # [MASK] token for padding dropped positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Fusion projection: concat(ft, g_pad) → fused hidden
        self.fusion_proj = nn.Linear(2 * hidden_size, hidden_size, bias=True)
        # --------------------------------------------------------
        # print("input_size:", input_size)
        kernel_size = patch_size
        self.x_embedder = PatchEmbed(
            input_height, input_width, patch_size, in_channels, hidden_size, kernel_size=kernel_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)  # timestep embedding type
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.y_embedder = CaptionEmbedder(in_channels=768, hidden_size=hidden_size, uncond_prob=0.1,
                                          act_layer=approx_gelu, token_num=self.contxt_len)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        blocks = []
        for i in range(depth):
            use_v1_residual = i > 0
            blocks.append(
                SiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    TT=self.contxt_len,
                    **block_kwargs,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.final_layer = FinalLayer(decoder_hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.x_embedder.grid_size, cls_token=0, extra_tokens=0
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        # nn.init.constant_(self.final_layer.linear_cls.weight, 0)
        # nn.init.constant_(self.final_layer.linear_cls.bias, 0)

    # def unpatchify(self, x, patch_size=None):
    #     """
    #     x: (N, T, patch_size**2 * C)
    #     imgs: (N, C, H, W)
    #     """
    #     c = self.out_channels
    #     p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
    #     h = w = int(x.shape[1] ** 0.5)
    #     assert h * w == x.shape[1]
    #
    #     x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    #     x = torch.einsum('nhwpqc->nchpwq', x)
    #     imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
    #     return imgs

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        T_p = 1
        H_p = W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        # unpad
        x = x[:, :, :R_t, :R_h, :R_w]
        return x

    # --------------------------- SPRINT helpers ---------------------------
    def _drop_tokens(self, x, TT, drop_ratio):
        """
        Randomly drop a fraction of tokens (except we ensure at least one token kept).

        x: (B, T, C)
        drop_ratio: fraction of tokens to drop (0.0 ~ 1.0)
        Returns:
            x_keep: (B, T_keep, C)
            ids_keep: (B, T_keep) indices into original T, or None if no drop.
        """
        if drop_ratio <= 0.0:
            return x, None

        B, T, C = x.shape
        T = T - TT
        if T <= 1:
            return x, None

        num_keep = max(1, int(T * (1.0 - drop_ratio)))
        if num_keep >= T:
            return x, None

        device = x.device
        noise = torch.rand(B, T, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]  # (B, T_keep)
        # print("ids_keep:", ids_keep.shape, ids_keep)
        x_keep_head = x[:, :TT, :]
        x_keep = x[:, TT:, :].gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, C))
        x_keep_out = torch.cat((x_keep_head, x_keep), dim=1)
        # print("x_keep_out:", x_keep_out.shape, x_keep_out)
        return x_keep_out, ids_keep

    def _pad_with_mask(self, x_sparse, ids_keep, TT, T_full):
        """
        x_sparse: (B, T_keep, C)
        ids_keep: (B, T_keep)
        TT:  text sequence length T
        T_full: full sequence length T
        Returns:
            x_pad: (B, T_full, C) with [MASK] at dropped positions.
        """
        if ids_keep is None:
            return x_sparse

        B, T_keep, C = x_sparse.shape
        assert T_full >= T_keep - TT
        # self.mask_token.register_hook(save_grad('dw:'))
        x_pad = self.mask_token.expand(B, T_full, C).clone()
        x_t = x_sparse[:, :TT, :]
        x_img = x_sparse[:, TT:, :]
        x_pad.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, C), x_img)
        out = torch.cat([x_t, x_pad], dim=1)
        return out

    def _sprint_fuse(self, f_dense, g_full):
        """
        f_dense: (B, T, C) encoder output ft
        g_full: (B, T, C) padded sparse output g_pad
        Returns fused h: (B, T, C)
        """
        h = torch.cat([f_dense, g_full], dim=-1)  # (B, T, 2C)
        # h.register_hook(save_grad('dh:'))
        h = self.fusion_proj(h)
        return h

    def get_dynamic_size(self, x):
        _, _, T, H, W = x.size()
        if H % self.patch_size != 0:
            H += self.patch_size - H % self.patch_size
        if W % self.patch_size != 0:
            W += self.patch_size - W % self.patch_size
        H = H // self.patch_size
        W = W // self.patch_size
        return (T, H, W)

    def _run_block(self, block, x, c, thw):
        if self.training and self.gradient_checkpointing:
            def custom_forward(hidden_states, cond):
                return block(hidden_states, cond, thw)
            return torch.utils.checkpoint.checkpoint(
                custom_forward,
                x,
                c,
                use_reentrant=False,
            )
        return block(x, c, thw)

    def _run_final_layer(self, x, c):
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                self.final_layer,
                x,
                c,
                use_reentrant=False,
            )
        return self.final_layer(x, c)

    # ---------------------------------------------------------------------
    def forward(self, x, t, y, uncond: bool = False):
        """
        Forward pass of SiT with SPRINT.
        x: (B, C, T, H, W) tensor of video latents
        t: (N,) tensor of diffusion timesteps
        y: (B, L, D) tensor of caption embeddings
        """

        B = x.shape[0]
        _, _, Tx, Hx, Wx = x.size()
        T, H, W = self.get_dynamic_size(x)
        S = H * W

        # Patch embedding
        x = rearrange(x, "B C T H W -> (B T) C H W")
        x = self.x_embedder(x)  # (B, N, D) 1152
        x = rearrange(x, "(B T) S C -> B (T S) C", B=B, T=T, S=S) # [B, 3 * 11 * 20, 1152] = [B, 660, 1152]

        # timestep and class embedding
        t_embed = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        c = t_embed

        # ------------------------------------------------------------------
        # 1) Encoder fθ on all tokens (dense, shallow)
        # ------------------------------------------------------------------
        x_enc = torch.cat([y, x], dim=1)  # [B, 77 + 660, 1152]

        for i in range(self.num_f):
            #x_enc = self._run_block(self.blocks[i], x_enc, c, (T, H, W))  # (N, T, D)
            x_enc = self.blocks[i](x_enc, c, TT=self.contxt_len, T=T, H=H, W=W, S=S)  # (N, T, D)

        # ------------------------------------------------------------------
        # 3) Middle blocks gθ on sparse tokens
        # ------------------------------------------------------------------
        x_mid = x_enc
        for i in range(self.num_f, self.num_f + self.num_g):
            #x_mid = self._run_block(self.blocks[i], x_mid, c, (T, H, W))  # (N, T_keep, D)
            x_mid = self.blocks[i](x_mid, c, TT=self.contxt_len, T=T, H=H, W=W, S=S)  # (N, T_keep, D)

        # ------------------------------------------------------------------
        # 4) Pad back to full length with [MASK] to get g_pad
        # ------------------------------------------------------------------
        g_pad = x_mid
        # ------------------------------------------------------------------
        # 5) Path-drop learning
        #    - During training: same behavior as before (stochastic path drop).
        #    - During sampling: enable path drop only for unconditional flow
        #      via the `uncond` flag.
        # ------------------------------------------------------------------
        if self.training and self.path_drop_prob > 0.0:
            # Sync random decision across all ranks
            drop_path = torch.rand(1, device=x.device)
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(drop_path, src=0)
            if drop_path.item() < self.path_drop_prob:
                # Keep gradient flow but zero out the contribution
                g_pad = g_pad * 0.0 + self.mask_token.expand_as(g_pad)
        elif uncond:  # Drop path for all samples
            g_pad = g_pad * 0.0 + self.mask_token.expand_as(g_pad)

        # ------------------------------------------------------------------
        # 6) Sparse–dense residual fusion: h_in = Fusion(ft, g_pad)
        # ------------------------------------------------------------------
        h_in = self._sprint_fuse(x_enc, g_pad)  # (N, T, D)

        # ------------------------------------------------------------------
        # 7) Decoder hθ on fused representation
        # ------------------------------------------------------------------
        x_dec = h_in
        for i in range(self.num_f + self.num_g, self.depth):
            #x_dec = self._run_block(self.blocks[i], x_dec, c, (T, H, W))
            x_dec = self.blocks[i](x_dec, c, TT=self.contxt_len, T=T, H=H, W=W, S=S)

        img_o = x_dec[:, self.contxt_len:, ...]

        x_out = self._run_final_layer(img_o, c)

        x_out = self.unpatchify(x_out, T, H, W, Tx, Hx, Wx)

        return x_out


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int or (grid_height, grid_width)
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, Iterable):
        grid_h_size, grid_w_size = grid_size
    else:
        grid_h_size = grid_size
        grid_w_size = grid_size

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_h_size, grid_w_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL_1(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=1, num_heads=16, encoder_depth=8,
               **kwargs)


def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, encoder_depth=8,
               **kwargs)


def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=4, num_heads=16, encoder_depth=8,
               **kwargs)


def SiT_L_1(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=1, num_heads=16, encoder_depth=8,
               **kwargs)


def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=2, num_heads=16, encoder_depth=8,
               **kwargs)


def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=4, num_heads=16, encoder_depth=8,
               **kwargs)


def SiT_B_1(**kwargs):
    return SiT(depth=16, hidden_size=1152, decoder_hidden_size=1152, patch_size=1, num_heads=16, encoder_depth=4,
               **kwargs)


def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, encoder_depth=4,
               **kwargs)


def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=4, num_heads=12, encoder_depth=4,
               **kwargs)


def SiT_S_1(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=1, num_heads=6, encoder_depth=4, **kwargs)


def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, encoder_depth=4, **kwargs)


def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, encoder_depth=4, **kwargs)


SiT_models = {
    'SiT-XL/1': SiT_XL_1, 'SiT-XL/2': SiT_XL_2, 'SiT-XL/4': SiT_XL_4,
    'SiT-L/1': SiT_L_1, 'SiT-L/2': SiT_L_2, 'SiT-L/4': SiT_L_4,
    'SiT-B/1': SiT_B_1, 'SiT-B/2': SiT_B_2, 'SiT-B/4': SiT_B_4,
    'SiT-S/1': SiT_S_1, 'SiT-S/2': SiT_S_2, 'SiT-S/4': SiT_S_4,
}

