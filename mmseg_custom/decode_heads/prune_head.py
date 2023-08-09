import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional
import math
from functools import partial
# from mmcv.runner import auto_fp16, force_fp32
import matplotlib.pyplot as plt

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from timm.models.layers import trunc_normal_
import matplotlib.pyplot as plt
from mmseg.models.losses import accuracy
from .atm_head import *


def convert_true_idx(orig, new):
    assert (~orig).sum() == len(
        new), "batch_idx and new pos mismatch!!! orig:{}, new:{} ".format((~orig).sum(), len(new))
    orig_new = torch.zeros_like(orig)
    orig_new[~orig] = new
    return orig_new


@HEADS.register_module()
class PruneHead(BaseDecodeHead):
    def __init__(
            self,
            img_size,
            in_channels,
            layers_per_decoder=2,
            num_heads=6,
            thresh=1.0,
            **kwargs,
    ):
        super(PruneHead, self).__init__(
            in_channels=in_channels, **kwargs)
        self.thresh = thresh
        self.image_size = img_size
        nhead = num_heads
        dim = self.channels
        proj = nn.Linear(self.in_channels, dim)
        trunc_normal_(proj.weight, std=.02)
        norm = nn.LayerNorm(dim)
        decoder_layer = TPN_DecoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * 4)
        decoder = TPN_Decoder(decoder_layer, layers_per_decoder)

        self.input_proj = proj
        self.proj_norm = norm
        self.decoder = decoder
        self.q = nn.Embedding(self.num_classes, dim)

        self.class_embed = nn.Linear(dim, 1 + 1)
        delattr(self, 'conv_seg')

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def per_img_forward(self, q, x):
        x = self.proj_norm(self.input_proj(x))
        q, attn = self.decoder(q, x.transpose(0, 1))
        cls = self.class_embed(q.transpose(0, 1))
        pred = cls.softmax(dim=-1)[..., :-1] * attn.sigmoid()
        return attn, cls, pred

    def forward(self, inputs, inference=False, batch_idx=None, canvas=None):
        if inference:
            # x = self._transform_inputs(inputs)
            x = inputs
            canvas_copy = canvas.clone()
            if x.dim() == 4:
                x = self.d4_to_d3(x)
            B, hw, ch = x.shape
            if batch_idx.sum():
                cls = []
                pred = []
                qs = self.q.weight.repeat(B, 1, 1).transpose(0, 1)
                for b in range(B):
                    q = qs[:, b].unsqueeze(1)
                    x_ = x[b][~batch_idx[b]]
                    attn_, cls_, pred_ = self.per_img_forward(q, x_.unsqueeze(0))
                    cls.append(cls_)
                    pred.append(pred_[0])
                    canvas_copy[b][:, ~batch_idx[b]] = attn_[0]

                cls = torch.cat(cls, dim=0)
                self.results = {"attn": canvas_copy}
                self.results.update({"pred_logits": cls})
                self.results.update({"pred": cls.softmax(
                    dim=-1)[..., :-1] * canvas_copy.sigmoid()})

            else:
                q = self.q.weight.repeat(B, 1, 1).transpose(0, 1) # q.shape [cls, b, ch]
                attn, cls, pred = self.per_img_forward(q, x) # x.shape [b, hw, ch]
                canvas_copy = attn
                self.results = {"attn": canvas_copy}
                self.results.update({"pred_logits": cls})
                self.results.update({"pred": pred})

            if self.thresh == 1:
                return batch_idx, canvas_copy
            else:
                with torch.no_grad():
                    for b, pred_ in enumerate(pred):
                        val, ind = pred_.max(dim=0)
                        pos = val > self.thresh
                        # keep top5 or smaller per class
                        for per_cls in ind[pos].unique():
                            per_cls_topk = (
                                ind[pos] == per_cls).sum().clamp_max(5)
                            topk_v, topk_idx = val[pos][ind[pos] == per_cls].topk(
                                per_cls_topk)
                            pos_idx = pos.nonzero()
                            ind_idx = pos_idx[ind[pos] == per_cls]
                            topk_idx = ind_idx[topk_idx]
                            pos[topk_idx] = False

                            # per_cls_idx = pos_idx[ind[pos] == per_cls]
                            # pos[per_cls_idx] = pos[per_cls_idx].index_fill_(
                            #     0, topk_idx, False)
                        pos_true = convert_true_idx(batch_idx[b], pos)
                        batch_idx[b] = torch.bitwise_or(
                            batch_idx[b], pos_true)

                return batch_idx, canvas_copy
        else:
            pred = self.results["pred"]
            pred = self.d3_to_d4(pred.transpose(-1, -2))
            pred = F.interpolate(pred, size=(self.image_size, self.image_size),
                                 mode='bilinear', align_corners=False)

            out = {"pred": pred}
            if self.training:
                out["pred_logits"] = self.results["pred_logits"]
                out["pred_masks"] = self.d3_to_d4(
                    self.results["attn"].transpose(-1, -2))
                return out
            else:
                return out["pred"]

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        return semseg

    def d3_to_d4(self, t):
        n, hw, c = t.size()
        if hw % 2 != 0:
            t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(n, c, h, w)

    def d4_to_d3(self, t):
        return t.flatten(-2).transpose(-1, -2)

    def loss_by_feat(self, seg_logit, batch_data_samples):
        seg_label = self._stack_batch_gt(batch_data_samples)
        # atm loss
        seg_label = seg_label.squeeze(1)
        loss = self.loss_decode(
            seg_logit,
            seg_label,
            ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(seg_logit["pred"], seg_label, ignore_index=self.ignore_index)
        return loss