import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
from typing import Optional, Tuple, Any
from typing import List, Optional, Tuple
# from utils import logger
from einops import rearrange
import math
import warnings
from .pos_embed import get_2d_sincos_pos_embed_new, get_1d_sincos_pos_embed_from_grid
from .deepseek_moe import MoEArgs, MoE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

__all__ = ["WiFo_CF", "Transformer"]

Tensor = torch.Tensor



class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()  # 5000,512
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)  # 1,5000
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()  # 512
        # position * div_term ： 5000 * 512
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # 1,5000,512
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x : 64,t_in,16
        return self.pe[:, :x.size(1)]  # 1，t_in


def scale_dot_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        dropout_p: float = 0.0,
        attn_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    _, _, E = q.shape
    q = q / math.sqrt(E)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = F.softmax(attn, dim=-1)
    if dropout_p:
        attn = F.dropout(attn, p=dropout_p)
    out = torch.bmm(attn, v)

    return out, attn


def multi_head_attention_forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Optional[Tensor],
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight=None,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    head_dim = embed_dim // num_heads
    q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"

        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len). \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    if not training:
        dropout_p = 0.0
    attn_output, attn_output_weights = scale_dot_attention(q, k, v, attn_mask, dropout_p)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)
    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


def _in_projection_packed(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w: Tensor,
        b: Optional[Tensor] = None,
) -> List[Tensor]:
    E = q.size(-1)
    if k is v:
        if q is k:
            return F.linear(q, w, b).chunk(3, dim=-1)
        else:
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True,
                 kdim=None, vdim=None, batch_first=False) -> None:
        # factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == self.embed_dim and self.vdim == self.embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim)))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim)))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim)))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim)))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=nn.ReLU(),
                 layer_norm_eps=1e-5, batch_first=False, norm=nn.LayerNorm,
                 moeargs=MoEArgs()) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead,
                                            dropout=dropout, batch_first=batch_first)
        self.norm1 = norm(d_model, eps=layer_norm_eps)
        self.norm2 = norm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

        self.activation = activation if isinstance(activation, nn.Module) else activation()

        self.moe = MoE(moeargs) if moeargs is not None else nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.is_moe = True if moeargs is not None else False

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if self.is_moe:
            src2, loss = self.moe(src, return_load_balance=True)
        else:
            src2 = self.moe(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if self.is_moe:
            return src, loss
        else:
            return src


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, is_moe=True, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layer = encoder_layer
        self.num_layers = num_layers
        self.norm = norm
        self.is_moe = is_moe

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src
        if self.is_moe:
            total_loss = 0
            for h in self.layer:
                output, loss = h(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
                total_loss += loss
            if self.norm is not None:
                output = self.norm(output)
            return output, total_loss
        else:
            for h in self.layer:
                output = h(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

            if self.norm is not None:
                output = self.norm(output)
            return output




# Decoder Layer:
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=nn.ReLU(),
                 layer_norm_eps=1e-5, batch_first=False, norm=nn.LayerNorm,
                 moeargs=MoEArgs()) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead,
                                            dropout=dropout, batch_first=batch_first)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        self.norm1 = norm(d_model, eps=layer_norm_eps)
        self.norm2 = norm(d_model, eps=layer_norm_eps)
        self.norm3 = norm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation if isinstance(activation, nn.Module) else activation()

        self.moe = MoE(moeargs) if moeargs is not None else nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.is_moe = True if moeargs is not None else False

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if self.is_moe:
            tgt2, loss = self.moe(tgt2, return_load_balance=True)
        else:
            tgt2 = self.moe(tgt2)

        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if self.is_moe:
            return tgt, loss
        else:
            return tgt


# Decoder
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, is_moe=True, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layer = decoder_layer
        self.num_layers = num_layers
        self.norm = norm
        self.is_moe = is_moe

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt
        if self.is_moe:
            total_loss = 0
            for h in self.layer:
                output, loss = h(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)
                total_loss += loss
            if self.norm is not None:
                output = self.norm(output)
            return output, total_loss
        else:
            for h in self.layer:
                output = h(output, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)
            if self.norm is not None:
                output = self.norm(output)
            return output


class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.)
        return self.conv(x)


class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class Transformer(nn.Module):

    def __init__(self, dim_patch: int = 64, dropout: float = 0.1,
                 d_model: int = 64, dim_feedforward: int = 64,
                 nhead: int = 8, num_encoder_layers: int = 2, num_decoder_layers: int = 4,
                 d_model_decoder: int = 64, dim_feedforward_decoder: int = 128,
                 if_moe_activate: bool = True, n_routed_experts: int = 16, n_activated_experts: int = 2, n_shared_expert: int = 1,
                 activation=nn.ReLU(), custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, reduction=64,
                 patch_size=[(4, 4), (4, 2), (2, 2)], beta1=0.1,  beta2=0.01, beta3=0.01,
                 feedback_type='uniform', num_bit=4) -> None:
        super(Transformer, self).__init__()
        print("Info: ", num_encoder_layers, num_decoder_layers, dim_patch, d_model, d_model_decoder, dim_feedforward, dim_feedforward_decoder,
              n_routed_experts, n_activated_experts, n_shared_expert, 'CR:', 64/(dim_patch/32))
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.if_moe_activate = if_moe_activate
        self.patch_size = patch_size
        self.reduction = reduction
        self.feedback_type = feedback_type
        self.num_bit = num_bit
        self.d_model_decoder = d_model_decoder
        self.csi_2d_patch_1 = nn.Conv2d(2, dim_patch, patch_size[0], patch_size[0], 0)
        self.csi_2d_patch_2 = nn.Conv2d(2, dim_patch, patch_size[1], patch_size[1], 0)
        self.csi_2d_patch_3 = nn.Conv2d(2, dim_patch, patch_size[2], patch_size[2], 0)
        self.proj_embed_encoder = nn.Linear(dim_patch, d_model)
        self.moeargs = MoEArgs()
        self.moeargs.dim = d_model
        self.moeargs.moe_inter_dim = dim_feedforward
        # moe
        self.moeargs.n_routed_experts = n_routed_experts
        self.moeargs.n_shared_experts = 0
        self.moeargs.n_activated_experts = n_activated_experts

        self.moeargs_d = MoEArgs()
        self.moeargs_d.dim = d_model_decoder
        self.moeargs_d.moe_inter_dim = dim_feedforward_decoder
        # moe
        self.moeargs_d.n_routed_experts = n_routed_experts
        self.moeargs_d.n_shared_experts = n_shared_expert
        self.moeargs_d.n_activated_experts = n_activated_experts
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            if if_moe_activate:
                encoder_layer = nn.ModuleList(
                    [TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                            activation, layer_norm_eps, batch_first,
                                            norm=RMSNorm, moeargs=self.moeargs) for i in range(num_encoder_layers)])
            else:
                encoder_layer = nn.ModuleList(
                    [TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                             activation, layer_norm_eps, batch_first,
                                             norm=RMSNorm, moeargs=None) for i in range(num_encoder_layers)])
            # encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
            encoder_norm = RMSNorm(dim=d_model, eps=layer_norm_eps)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, norm=encoder_norm, is_moe=if_moe_activate)
        self.proj_output_encoder = nn.Linear(d_model, dim_patch)

        self.cnn_compression = nn.Sequential(
            nn.Conv2d(1, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            DownSampleBlock(64),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            DownSampleBlock(64),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            DownSampleBlock(64),
            nn.Conv2d(64, 1, 5, 1, 2),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.cnn_expansion = nn.Sequential(
            nn.Conv2d(1, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            UpSampleBlock(64),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            UpSampleBlock(64),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            UpSampleBlock(64),
            nn.Conv2d(64, 1, 5, 1, 2),
            nn.BatchNorm2d(1),
        )
        self.proj_embed_decoder = nn.Linear(dim_patch, d_model_decoder)
        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            if if_moe_activate:
                decoder_layer = nn.ModuleList(
                    [TransformerDecoderLayer(d_model_decoder, nhead, dim_feedforward_decoder * n_shared_expert, dropout,
                                     activation, layer_norm_eps, batch_first,
                                     norm=RMSNorm, moeargs=self.moeargs_d) for i in range(num_decoder_layers)])
            else:
                decoder_layer = nn.ModuleList(
                    [TransformerDecoderLayer(d_model_decoder, nhead, dim_feedforward_decoder * n_shared_expert, dropout,
                                             activation, layer_norm_eps, batch_first,
                                             norm=RMSNorm, moeargs=None) for i in range(num_decoder_layers)])
            # decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
            decoder_norm = RMSNorm(dim=d_model_decoder, eps=layer_norm_eps)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, norm=decoder_norm, is_moe=if_moe_activate)
        self.embed_proj_bs_1 = nn.Linear(d_model_decoder, patch_size[0][0] * patch_size[0][1] * 2)
        self.embed_proj_bs_2 = nn.Linear(d_model_decoder, patch_size[1][0] * patch_size[1][1] * 2)
        self.embed_proj_bs_3 = nn.Linear(d_model_decoder, patch_size[2][0] * patch_size[2][1] * 2)
        self.d_model = d_model

        self.nhead = nhead

        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        self.encoder_names = ["csi_2d_patch", "encoder", "cnn_compression", "embed_proj_ue"]
        self.decoder_names = ["proj_embed_decoder", "cnn_expansion", "decoder", "embed_proj_bs"]
        self.trainable_names =  ["csi_2d_patch", "embed_proj_bs", "cnn_compression", "cnn_expansion"]

        self._reset_parameters()

    def forward(self, src: Tensor, tgt: Optional[Tensor] = None, src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                quan_scheme='mu', bit_user: Optional[Tensor] = None,
                dataset_type=1) -> tuple[Any, Tensor]:
        # parms
        in_size = src.shape[-2:]  # N, K

        nu = src.shape[2]
        if bit_user is None:
            bit_user = torch.ones([nu]) * 4
        # UE side
        csi_reshape = rearrange(src, 'b o u n k -> (b u) o n k')  # 'u' mean num of user
        csi_embed_ue = self.csi_2d_patch_1(csi_reshape)  # (b u), dim_patch, n', k'  [n' = n // p]
        grid_size = [in_size[0] // self.patch_size[dataset_type-1][0], in_size[1] // self.patch_size[dataset_type-1][1]]
        L = (in_size[0] * in_size[1]) // (self.patch_size[dataset_type-1][0] * self.patch_size[dataset_type-1][1])
        csi_embed_ue = rearrange(csi_embed_ue, '(b u) d n k -> b (u n k) d', u=nu)  # (b u), (n' k'), dim_patch  [n' = n // p]
        csi_embed_ue = self.proj_embed_encoder(csi_embed_ue)  # (b u), (n' k'), d_model
        csi_embed_ue = self.get_user_embed(csi_embed_ue, nu=nu, bit_user=bit_user, grid_size=grid_size)
        csi_embed_ue = rearrange(csi_embed_ue, 'b (u m) d ->(b u) m d', u=nu)
        if self.if_moe_activate:
            csi_embed_ue, loss_encoder = self.encoder(csi_embed_ue, mask=src_mask,
                                    src_key_padding_mask=src_key_padding_mask)  # (b u), n, d_model
        else:
            csi_embed_ue = self.encoder(csi_embed_ue, mask=src_mask,
                                    src_key_padding_mask=src_key_padding_mask)  # (b u), n, d_model
        csi_embed_ue = self.proj_output_encoder(csi_embed_ue)
        # Compress
        csi_compress = self.cnn_compression(csi_embed_ue.unsqueeze(1))  # (b u), 1, 8, 8
        # Dynamic Quan Network
        csi_compress = rearrange(csi_compress, '(b u) o n k -> b u (o n k)', u=nu)
        csi_quan = self.quantize_vector(csi_compress, bit_user, quan_scheme=quan_scheme)
        # preserve gradients
        csi_quan = csi_compress + (csi_quan - csi_compress).detach()  # moving average instead of hard codebook remapping
        loss_quan = torch.mean((csi_quan.detach() - csi_compress) ** 2)
        # BS side
        csi_quan = rearrange(csi_quan, 'b u (o n k) -> (b u) o n k', u=nu, o=1, n=L // 8)
        csi_expansion = self.cnn_expansion(csi_quan)  # (b u), (n' k'), d_model
        csi_expansion = rearrange(csi_expansion, '(b u) o n k -> b (u n) (o k)', u=nu)  # b, (u n), (o nc)
        csi_expansion = self.proj_embed_decoder(csi_expansion)
        csi_expansion = self.get_user_embed(csi_expansion, nu=nu, bit_user=bit_user, grid_size=grid_size)
        if self.if_moe_activate:
            output, loss_decoder = self.decoder(csi_expansion, csi_expansion, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
        else:
            output = self.decoder(csi_expansion, csi_expansion, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
        if dataset_type == 1:
            output = self.embed_proj_bs_1(output)  # b, n, k
        elif dataset_type == 2:
            output = self.embed_proj_bs_2(output)  # b, n, k
        else:
            output = self.embed_proj_bs_3(output)  # b, n, k
        output = rearrange(output, 'b (u n) k -> (b u) n k', u=nu)
        output = self.fold2d(output, output_size=in_size, patch_size=self.patch_size[dataset_type-1])
        output = rearrange(output, '(b u) o n k -> b o u n k', u=nu)
        return output, self.beta1 * loss_quan, self.beta2*loss_encoder, self.beta3*loss_decoder

    def get_user_embed(self, x, nu, grid_size, bit_user=None):
        dim = x.shape[-1]
        L = x.shape[1] // nu
        bit_embed = torch.from_numpy(get_1d_sincos_pos_embed_from_grid(dim // 2, bit_user.cpu().repeat(1, L).view(-1))).to(x.device).to(
            x.dtype)
        bit_embed.requires_grad_(False)

        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed_new(dim // 2, grid_size, False)).repeat(nu, 1).to(
            x.device).to(x.dtype)
        pos_embed.requires_grad_(False)
        pos_embed = torch.cat([pos_embed, bit_embed], dim=1).requires_grad_(False)

        return x + pos_embed.unsqueeze(0)

    def mu_law_quantize(self, x, num_bits, mu=255):
        x_mu = torch.sign(x) * torch.log1p(mu * x) / torch.log1p(torch.tensor(mu, dtype=x.dtype))
        levels = 2 ** num_bits.to(x.device)
        x_q = torch.ceil(x_mu * levels) / levels
        x_dequant = torch.sign(x_q) * (torch.expm1(torch.abs(x_q) * torch.log1p(torch.tensor(mu, dtype=x.dtype))) / mu)
        return x_dequant

    def uniform_quantize(self, x, num_bits):
        levels = 2 ** num_bits.to(x.device)
        x_q = torch.ceil(x * levels) / levels
        return x_q

    def quantize_vector(self, X, bit_widths, quan_scheme='uniform'):
        B, Nu, L = X.shape
        bit_widths = bit_widths.view(1, Nu, 1)
        if quan_scheme == 'uniform':
            X_q = self.uniform_quantize(X, bit_widths)
        elif quan_scheme == 'mu':
            X_q = self.mu_law_quantize(X, bit_widths)
        return X_q

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:

        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)

        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def zero_pad_tensor(self, x: torch.Tensor, P: int) -> torch.Tensor:
        x_padded = x.clone()
        x_padded[:, -P:] = 0
        return x_padded

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by setting a random proportion of elements to zero or random values.
        x: [N, L, D], sequence
        Returns:
            x_masked: The input x with masked values replaced by zero or random values.
        """
        N, L, D = x.shape  # batch, length, dim
        len_mask = int(L * mask_ratio)  # Number of elements to mask

        # Generate random indices for masking
        mask_indices = torch.rand(N, L, device=x.device).argsort(dim=1)[:, :len_mask]

        # Create a mask tensor initialized to ones (1 means keep, 0 means mask)
        mask = torch.ones((N, L), device=x.device)
        mask.scatter_(1, mask_indices, 0)  # Set masked positions to 0

        # Expand the mask to match the dimensions of x
        mask = mask.unsqueeze(-1).expand(-1, -1, D)

        # Apply the mask to x: masked positions set to zero/random values
        x_masked = x * mask + (1 - mask) * torch.rand_like(x)  # Zero out masked values or replace with random values

        return x_masked

    def fold2d(self, token, output_size, patch_size):
        # token: B, (T/p, N/p), p**2
        # output_csi: B, T, N
        B, L, D = token.shape
        assert D == 2 * patch_size[0] * patch_size[1]
        K, N = output_size
        assert K % patch_size[0] == 0 and N % patch_size[1] == 0
        T_p, N_p = K // patch_size[0], N // patch_size[1]

        # Reshape and permute the token to match the desired output structure
        token_seq = token.view(B, T_p, N_p, 2, patch_size[0], patch_size[1])
        output_csi = token_seq.permute(0, 3, 1, 4, 2, 5)  # Rearrange dimensions
        output_csi = output_csi.reshape(B, 2, K, N)  # Merge patch dimensions back

        return output_csi


def WiFo_CF(args):
    r""" Create a proposed WiFo-CF.
    """
    model = Transformer(dim_patch=args.dim_patch,
                        d_model=args.d_model,
                        d_model_decoder=args.d_model_decoder,
                        dim_feedforward=args.d_ff,
                        dim_feedforward_decoder=args.d_ff_d,
                        num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_decoder_layers,
                        nhead=8, reduction=args.cr,
                        n_routed_experts=args.n_routed_experts, n_activated_experts=args.n_activated_experts,
                        n_shared_expert=args.n_shared_expert,
                        feedback_type=args.feedback_type,
                        num_bit=args.num_bit,
                        if_moe_activate=args.if_moe_activate,
                        beta1=args.beta1, beta2=args.beta2, beta3=args.beta3,
                        dropout=0.05)
    return model


