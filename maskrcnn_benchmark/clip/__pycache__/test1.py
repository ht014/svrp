# uncompyle6 version 3.7.4
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.11 (default, Jul 27 2021, 14:32:16) 
# [GCC 7.5.0]
# Embedded file name: /home/hetao/CLIP-1.7/clip/model.py
# Compiled at: 2021-09-18 09:57:03
# Size of source mod 2**32: 21788 bytes
from collections import OrderedDict
from typing import Tuple, Union
import numpy as np, torch
import torch.nn.functional as F
from torch import nn
from maskrcnn_benchmark.layers import ROIAlign
from maskrcnn_benchmark.modeling.poolers import Pooler

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, (planes * self.expansion), 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
             (
              '-1', nn.AvgPool2d(stride)),
             (
              '0', nn.Conv2d(inplanes, (planes * self.expansion), 1, stride=1, bias=False)),
             (
              '1', nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: torch.Tensor):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ModifiedBottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x: torch.Tensor):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):

    def __init__(self, spacial_dim, embed_dim, num_heads, output_dim=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.output_dim = output_dim

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(query=x,
          key=x,
          value=x,
          embed_dim_to_check=(x.shape[(-1)]),
          num_heads=(self.num_heads),
          q_proj_weight=(self.q_proj.weight),
          k_proj_weight=(self.k_proj.weight),
          v_proj_weight=(self.v_proj.weight),
          in_proj_weight=None,
          in_proj_bias=(torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias])),
          bias_k=None,
          bias_v=None,
          add_zero_attn=False,
          dropout_p=0,
          out_proj_weight=(self.c_proj.weight),
          out_proj_bias=(self.c_proj.bias),
          use_separate_proj_weight=True,
          training=(self.training),
          need_weights=False)
        return x


class ModifiedResNet(nn.Module):
    __doc__ = '\n    A ResNet class that is similar to torchvision\'s but contains the following changes:\n    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.\n    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1\n    - The final pooling layer is a QKV attention instead of an average pool\n    '

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, cfg=None):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, (width // 2), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d((width // 2), (width // 2), kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d((width // 2), width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self._inplanes = width
        self.trans_inplanes = width * 4
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer((width * 2), (layers[1]), stride=2)
        self.layer3 = self._make_layer((width * 4), (layers[2]), stride=2)
        self.layer4 = self._make_layer((width * 8), (layers[3]), stride=2)
        self.pooler = ROIAlign((7, 7),
          spatial_scale=0.0625, sampling_ratio=0)
        embed_dim = width * 32
        self.is_use_mult_att = cfg.MODEL.USE_MULT_ATTN
        self.pooler = Pooler(output_size=(7, 7),
          scales=(0.25, 0.125, 0.0625, 0.03125),
          sampling_ratio=2,
          is_use_mult_att=(self.is_use_mult_att))
        if self.is_use_mult_att:
            self.att1 = AttentionPool2d(7, 256, heads, output_dim)
            self.att2 = AttentionPool2d(7, 512, heads, output_dim)
            self.att3 = AttentionPool2d(7, 1024, heads, output_dim)
            self.att4 = AttentionPool2d(7, 2048, heads, output_dim)
            attnpools = [self.att1, self.att2, self.att3, self.att4]
            self.pooler.set_mutli_attnpools(attnpools)
            self.trans1 = self._make_trans_layer((width * 4), 3, stride=1)
            self.trans2 = self._make_trans_layer((width * 8), 2, stride=1)
            self.trans3 = self._make_trans_layer((width * 16), 2, stride=1)
            self.trans4 = self._make_trans_layer((width * 32), 1, stride=1)
        else:
            self.attnpool2 = AttentionPool2d(input_resolution // 32, 256, heads, output_dim)
            self.conv11 = nn.Conv2d(512, 256, 1, padding=0, bias=False)
            self.bn11 = nn.BatchNorm2d(256)
            self.conv22 = nn.Conv2d(1024, 256, 1, padding=0, bias=False)
            self.bn22 = nn.BatchNorm2d(256)
            self.conv33 = nn.Conv2d(2048, 256, 1, padding=0, bias=False)
            self.bn33 = nn.BatchNorm2d(256)

    def _make_trans_layer(self, planes, blocks, stride=1):
        layers = [ModifiedBottleneck(planes, planes, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ModifiedBottleneck(planes, planes, stride=stride))

        return (nn.Sequential)(*layers)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [
         Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return (nn.Sequential)(*layers)

    def forward(self, x, boxes):

        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))

            x = self.avgpool(x)
            return x

        feature_maps = []
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        if not self.is_use_mult_att:
            feature_maps.append(x)
        else:
            trans_x1 = self.trans1(x)
            feature_maps.append(trans_x1)
        x = self.layer2(x)
        if not self.is_use_mult_att:
            x11 = self.bn11(self.conv11(x))
            feature_maps.append(x11)
        else:
            trans_x2 = self.trans2(x)
            feature_maps.append(trans_x2)
        x = self.layer3(x)
        if not self.is_use_mult_att:
            x22 = self.bn22(self.conv22(x))
            feature_maps.append(x22)
        else:
            trans_x3 = self.trans3(x)
            feature_maps.append(trans_x3)
        x = self.layer4(x)
        if not self.is_use_mult_att:
            x33 = self.bn33(self.conv33(x))
            feature_maps.append(x33)
        else:
            trans_x4 = self.trans4(x)
            feature_maps.append(trans_x4)
        if not self.is_use_mult_att:
            roi_feats = self.pooler(feature_maps, boxes)
            x = self.attnpool2(roi_feats)
            return x[0]
        x = self.pooler(feature_maps, boxes)
        return x


class LayerNorm(nn.LayerNorm):
    __doc__ = "Subclass torch's LayerNorm to handle fp16."

    def forward(self, x):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
         ('c_fc', nn.Linear(d_model, d_model * 4)),
         ('gelu', QuickGELU()), ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=(x.dtype), device=(x.device)) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=(self.attn_mask))[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):

    def __init__(self, width, layers, heads, attn_mask=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = (nn.Sequential)(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):

    def __init__(self, input_resolution, patch_size, width, layers, heads, output_dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** (-0.5)
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros((x.shape[0]), 1, (x.shape[(-1)]), dtype=(x.dtype), device=(x.device)), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x


class CLIP(nn.Module):

    def __init__(self, embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size, context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, cfg):
        super().__init__()
        self.context_length = context_length
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(layers=vision_layers,
              output_dim=embed_dim,
              heads=vision_heads,
              input_resolution=image_resolution,
              width=vision_width,
              cfg=cfg)
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(input_resolution=image_resolution,
              patch_size=vision_patch_size,
              width=vision_width,
              layers=vision_layers,
              heads=vision_heads,
              output_dim=embed_dim)
        self.transformer = Transformer(width=transformer_width,
          layers=transformer_layers,
          heads=transformer_heads,
          attn_mask=(self.build_attention_mask()))
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim).float()).float()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(14.285714285714285))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_((self.token_embedding.weight), std=0.02)
        nn.init.normal_((self.positional_embedding), std=0.01)
        if isinstance(self.visual, ModifiedResNet):
            if not self.visual.is_use_mult_att:
                std = self.visual.attnpool2.c_proj.in_features ** (-0.5)
                nn.init.normal_((self.visual.attnpool2.q_proj.weight), std=std)
                nn.init.normal_((self.visual.attnpool2.k_proj.weight), std=std)
                nn.init.normal_((self.visual.attnpool2.v_proj.weight), std=std)
                nn.init.normal_((self.visual.attnpool2.c_proj.weight), std=std)
            else:
                for att in self.visual.pooler.mult_attns:
                    std = att.c_proj.in_features ** (-0.5)
                    nn.init.normal_((att.q_proj.weight), std=std)
                    nn.init.normal_((att.k_proj.weight), std=std)
                    nn.init.normal_((att.v_proj.weight), std=std)
                    nn.init.normal_((att.c_proj.weight), std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith('bn3.weight'):
                        nn.init.zeros_(param)

        proj_std = self.transformer.width ** (-0.5) * (2 * self.transformer.layers) ** (-0.5)
        attn_std = self.transformer.width ** (-0.5)
        fc_std = (2 * self.transformer.width) ** (-0.5)
        for block in self.transformer.resblocks:
            nn.init.normal_((block.attn.in_proj_weight), std=attn_std)
            nn.init.normal_((block.attn.out_proj.weight), std=proj_std)
            nn.init.normal_((block.mlp.c_fc.weight), std=fc_std)
            nn.init.normal_((block.mlp.c_proj.weight), std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_((self.text_projection), std=(self.transformer.width ** (-0.5)))

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, boxes):
        return self.visual(image.type(self.dtype), boxes)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[(torch.arange(x.shape[0]), text.argmax(dim=(-1)))] @ self.text_projection
        return x

    def forward(self, image, boxes, text):
        image_features = self.encode_image(image, boxes)
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=(-1), keepdim=True)
        text_features = text_features / text_features.norm(dim=(-1), keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return (
         logits_per_image, logits_per_text)


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()
        if isinstance(l, nn.MultiheadAttention):
            for attr in [
             *[f"{s}_proj_weight" for s in ('in', 'q', 'k', 'v')], 'in_proj_bias', 'bias_k', 'bias_v']:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ('text_projection', 'proj'):
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, cfg=None):
    vit = 'visual.proj' in state_dict
    if vit:
        vision_width = state_dict['visual.conv1.weight'].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith('visual.') if k.endswith('.attn.in_proj_weight')])
        vision_patch_size = state_dict['visual.conv1.weight'].shape[(-1)]
        grid_size = round((state_dict['visual.positional_embedding'].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts = [len(set((k.split('.')[2] for k in state_dict if k.startswith(f"visual.layer{b}")))) for b in (1,
                                                                                                                2,
                                                                                                                3,
                                                                                                                4)]
        vision_layers = tuple(counts)
        vision_width = state_dict['visual.layer1.0.conv1.weight'].shape[0]
        output_width = round((state_dict['visual.attnpool.positional_embedding'].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict['visual.attnpool.positional_embedding'].shape[0]
        image_resolution = output_width * 32
    embed_dim = state_dict['text_projection'].shape[1]
    context_length = state_dict['positional_embedding'].shape[0]
    vocab_size = state_dict['token_embedding.weight'].shape[0]
    transformer_width = state_dict['ln_final.weight'].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set((k.split('.')[2] for k in state_dict if k.startswith('transformer.resblocks'))))
    model = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size, context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, cfg)
    for key in ('input_resolution', 'context_length', 'vocab_size'):
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict, strict=False)
    return model.eval()

