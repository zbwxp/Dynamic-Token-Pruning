from .vit import VisionTransformer
from mmseg.registry import MODELS
import torch

@MODELS.register_module()
class ViT_prune(VisionTransformer):
    def __init__(self,
                num_classes,
                freeze=False,
                **kwargs,
                 ):
        super(ViT_prune, self).__init__(
            **kwargs,
        )
        self.num_classes = num_classes

    def forward(self, inputs, model):
        B = inputs.shape[0]

        x, hw_shape = self.patch_embed(inputs)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embeding(x, hw_shape, self.pos_embed)
        
        hw = hw_shape[0]*hw_shape[1]
        batch_idx = torch.zeros(
            (B, hw), device=x.device) != 0
        canvas = torch.zeros_like(batch_idx).unsqueeze(
            1).repeat(1, self.num_classes, 1).float()
        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]
        else:
            batch_idx = torch.cat(
                (torch.zeros_like(batch_idx[:, 0:1]), batch_idx), dim=1)

        outs = []
        for i, layer in enumerate(self.layers):
            # total += (~batch_idx).sum()
            if batch_idx[:, -hw:].sum() == 0:
                x = layer(x)
            else:
                x = layer(x, batch_idx)

            if i in self.out_indices:
                idx = self.out_indices.index(i)
                if i != self.out_indices[-1]:
                    batch_idx, canvas = model.auxiliary_head[idx](
                        x, inference=True, batch_idx=batch_idx, canvas=canvas)
                else:
                    batch_idx, canvas = model.decode_head(
                        x, inference=True, batch_idx=batch_idx, canvas=canvas)
                    if self.final_norm:
                        x = self.norm1(x)

                if self.with_cls_token:
                    # Remove class token and reshape token for decoder head
                    out = x[:, 1:]
                else:
                    out = x

                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1],
                                  C).permute(0, 3, 1, 2).contiguous()
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                outs.append(out)


        return tuple(outs)