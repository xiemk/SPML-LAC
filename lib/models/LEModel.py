import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math


from models.backbone import build_backbone
from models.encoder import build_encoder
from utils.helper import clean_state_dict

NUM_CHANNEL = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
}

NUM_CLASS = {
    'voc': 20,
    'coco': 80,
    'nus': 81,
    'cub': 312,
}

class GroupWiseLinear(nn.Module):

    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class LEModel(nn.Module):
    def __init__(self, backbone, encoder, num_class, feat_dim, is_proj):
        """[summary]
    
        Args:
            backbone : backbone model.
            encoder : encoder model.
            num_class : number of classes. (80 for MSCOCO).
            feat_dim : dimension of features.
            is_proj : open/close a projector.
        """
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.is_proj = is_proj
        
        hidden_dim = encoder.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_class, hidden_dim)
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)
        if is_proj:
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, feat_dim)
            )


    def forward(self, input):
        # import ipdb; ipdb.set_trace()
        src, pos = self.backbone(input)
        src, pos = src[-1], pos[-1]

        query_input = self.query_embed.weight
        hs = self.encoder(self.input_proj(src), query_input, pos)[0]    # B,K,d

        out = self.fc(hs[-1])


        if self.is_proj:
            feat = hs[-1]
            batch_size = feat.shape[0]
            feat = torch.cat(torch.unbind(feat, dim=0))                     #  == (batch_size, num_class, ...)->(batch_size * num_class, ...)
            feat = F.normalize(self.proj(feat), dim=1)                      #  == (..., hidden_dim)->(..., feat_dim)
            feat = torch.stack(torch.chunk(feat, batch_size, dim=0), dim=0) #  == (batch_size * num_class, ...)->(batch_size, num_class, ...)
            return out, feat
        else:
            return out


    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(), self.query_embed.parameters())

    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device(dist.get_rank()))
        # import ipdb; ipdb.set_trace()
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(path, checkpoint['epoch']))


def build_LEModel(args):
    args.num_class = NUM_CLASS[args.dataset_name]
    args.hidden_dim = NUM_CHANNEL[args.backbone]

    backbone = build_backbone(args)
    encoder = build_encoder(args)

    model = LEModel(
        backbone = backbone,
        encoder = encoder,
        num_class = args.num_class,
        feat_dim = args.feat_dim,
        is_proj = args.is_proj,
    )
    
    model.input_proj = nn.Identity()
    print("set model.input_proj to Indentify!")

    return model