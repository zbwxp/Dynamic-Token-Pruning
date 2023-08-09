# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from mmcv.cnn import ConvModule
import math

from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.utils import ConfigType, SampleList
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize

import matplotlib.pyplot as plt
from einops import rearrange, reduce, repeat

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@MODELS.register_module()
class MLPHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_layers=3,
                 mlp_ratio=4,
                 img_size=518,
                 patch_size=14,
                 **kwargs):
        super(MLPHead, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.img_size = img_size
        # del self.conv_seg
        self.layer = MLP(self.in_channels, int(self.channels * mlp_ratio), self.num_classes, num_layers)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        feats = self.layer(self.d4_to_d3(x) if x.ndim == 4 else x)
        return feats

    def forward(self, inputs):
        """Forward function."""
        x = self._forward_feature(inputs)
        if hasattr(self, "crop_idx") and self.crop_idx is not None:
            return repeat(x, 'b l c -> b c (l pad)', pad=self.patch_size**2)
        return self.d3_to_d4(x)
    
    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        if hasattr(self, "crop_idx") and self.crop_idx is not None:
            crop_label = rearrange(seg_label, 'b (we wp) (he hp) -> b (we he) (wp hp)', wp=self.patch_size, hp=self.patch_size)[:, self.crop_idx]
            seg_label = crop_label.flatten(-2)
        else:
            seg_logits = resize(
                input=seg_logits,
                size=seg_label.shape[1:],
                mode='bilinear',
                align_corners=self.align_corners)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss

    def d4_to_d3(self, t):
        return t.flatten(-2).transpose(-1, -2)
    
    def d3_to_d4(self, t):
        n, hw, c = t.size()
        # if hw % 2 != 0:
            # t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(n, c, h, w)