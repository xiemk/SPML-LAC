import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from torch.functional import Tensor
import math


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_layers: Dict,
    # return_interm_layers: bool
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, input: Tensor):
        xs = self.body(input)
        out: Dict[str, Tensor] = {}
        for name, x in xs.items():
            out[name] = x
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool = True,
                 return_interm_layers: bool = False,
                 dilation: bool = False):
        if name in ['resnet18', 'resnet50', 'resnet34', 'resnet101']:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=True,
                norm_layer=FrozenBatchNorm2d)
            if return_interm_layers:
                return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            else:
                return_layers = {'layer4': "0"}
        else:
            raise NotImplementedError("Unknow name: %s" % name)
        
        NUM_CHANNEL = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
            'resnet101': 2048,
        }
        
        num_channels = NUM_CHANNEL[name]
        super().__init__(backbone, train_backbone, num_channels, return_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, input: Tensor):
        xs = self[0](input)
        out: List[Tensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    
    backbone = Backbone(args.backbone)

    model = Joiner(backbone, position_embedding)

    model.num_channels = backbone.num_channels
    return model


################################################################################################################################


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, maxH=30, maxW=30):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.maxH = maxH
        self.maxW = maxW
        pe = self._gen_pos_buffer()
        self.register_buffer('pe', pe)

    def _gen_pos_buffer(self):
        _eyes = torch.ones((1, self.maxH, self.maxW))
        y_embed = _eyes.cumsum(1, dtype=torch.float32)
        x_embed = _eyes.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (torch.div(dim_t, 2, rounding_mode='floor')) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward(self, input: Tensor):
        x = input
        return self.pe.repeat((x.size(0),1,1,1))


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2

    downsample_ratio = 32
    # TODO find a better way of exposing other arguments
    assert args.img_size % 32 == 0, "args.img_size ({}) % 32 != 0".format(args.img_size)
    position_embedding = PositionEmbeddingSine(N_steps, normalize=True, maxH=args.img_size // downsample_ratio, maxW=args.img_size // downsample_ratio)


    return position_embedding


################################################################################################################################
