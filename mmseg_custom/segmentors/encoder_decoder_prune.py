from typing import Any, Optional, List
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder as BaseEncoderDecoder
from fvcore.nn import FlopCountAnalysis
from mmseg.structures import SegDataSample
from mmseg.utils import (ForwardResults, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList)
import torch
import torch.nn.functional as F
from torch import Tensor


from mmseg.models.utils import resize


@SEGMENTORS.register_module()
class EncoderDecoderPrune(BaseEncoderDecoder):
    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs, self)
        if self.with_neck:
            x = self.neck(x)
        return x