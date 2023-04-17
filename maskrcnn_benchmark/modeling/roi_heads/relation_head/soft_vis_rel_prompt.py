# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random
import torch
from torch import nn
import numpy as np
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from ..attribute_head.roi_attribute_feature_extractors import make_roi_attribute_feature_extractor
from ..box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_relation_feature_extractors import make_roi_relation_feature_extractor
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_union
from .loss import make_roi_relation_loss_evaluator
from .pretrain_transformer import TransformerContext
from maskrcnn_benchmark.clip import clip
from .text_transformer import TextTransformer
from copy import deepcopy
from maskrcnn_benchmark.data import get_dataset_statistics

def bbox_overlap(box_a, box_b):
    inter = bbox_intersection(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0] + 1.0) *
              (box_a[:, 3] - box_a[:, 1] + 1.0)).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0] + 1.0) *
              (box_b[:, 3] - box_b[:, 1] + 1.0)).unsqueeze(0).expand_as(inter)  # [A,B]
    union = torch.min(area_a,area_b)#area_b+area_a - inter #
    return inter / (union + 1e-9)


def bbox_intersection(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy + 1.0), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def sample_union_proposals(proposals, rel_pair_indx, per_image_rois, context_length= 6):
    sampled_proposals = []
    resort_rois = []
    sorted_proposals = []
    obj_labels = proposals.get_field('labels')
    subs = []
    objs = []
    for pair in rel_pair_indx:
        sub = pair[0]
        obj = pair[1]
        subs.append(obj_labels[sub])
        objs.append(obj_labels[obj])
        union_region_proposals = [] #[sub, obj]

        subject_box = proposals[torch.tensor([sub]).long().to(sub.device)]
        object_box =  proposals[torch.tensor([obj]).long().to(obj.device)]
        union_box =  boxlist_union(subject_box, object_box).bbox
        # print(union_box.shape,proposals.bbox.shape)
        # input()
        intersections = bbox_overlap(union_box,proposals.bbox)

        for i in range(intersections.shape[-1]):
            if intersections[0][i] < 0.95 or i == sub or i == obj: continue

            union_region_proposals.append(i)
            if len(union_region_proposals) > context_length:
                break

        if sub.item() not in union_region_proposals:
            union_region_proposals = [sub.item()] + union_region_proposals
        if obj.item() not in union_region_proposals:
            union_region_proposals.append(obj.item())
        # print("union_region_proposals length:",len(union_region_proposals))

        resort_rois.append(per_image_rois[union_region_proposals])
        boxes = proposals[torch.tensor(union_region_proposals).long().to(proposals.bbox.device)]
        sorted_proposals.append(boxes)
        sampled_proposals.append(union_region_proposals)

    return sampled_proposals,torch.cat(resort_rois,0),sorted_proposals, subs, objs


class SVRP(nn.Module):

    def __init__(self, cfg, in_channels):
        super(SVRP, self).__init__()

        self.cfg = cfg.clone()
        self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.context_layer = TransformerContext(cfg, in_channels)
        self.visual2text_decoder = TransformerContext(cfg, in_channels)
        statistics = get_dataset_statistics(cfg)
        self.obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
        'att_classes']
        # parameters
        self.text_transformer = TextTransformer(context_length=12)
        self.vis2tex_lin = nn.Linear(512, self.text_transformer.transformer_width)
        self.mask_embedding = nn.Embedding(1, 768)
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.obj_semantics = None
        self.lin_semantics = nn.Linear(512, self.text_transformer.transformer_width)

    def get_obj_semantics(self,device):
        if self.obj_semantics is not None:
            return
        self.obj_classes [0] = "background"
        tokens = clip.tokenize(self.obj_classes, context_length=12, truncate=True).long().to(device)
        self.obj_semantics = self.lin_semantics(self.text_transformer(tokens,norm = True))


    def forward(self, features, proposals, rel_pair_idxs):
        roi_features = self.box_feature_extractor(features, proposals)
        with torch.no_grad():
          self.get_obj_semantics(roi_features.device)
        image_features = self.context_layer(roi_features, proposals)

        num_objs = [len(p) for p in proposals]
        per_image_rois = image_features.split(num_objs, dim=0)
        all_proposals = []
        all_rois = []
        ss = []
        oo = []
        for p,rel_pair_idx,roi in zip(proposals,rel_pair_idxs,per_image_rois):
            _, sorted_rois, sorted_proposals,subs,objs = sample_union_proposals(p, rel_pair_idx, roi)
            all_proposals += sorted_proposals
            all_rois.append(sorted_rois)
            ss += subs
            oo += objs

        image_features = torch.cat(all_rois)

        # image_features = self.context_layer(roi_features, all_proposals)
        decode_vis2tex = self.visual2text_decoder(image_features, all_proposals, True)
        vis2tex_tokens = self.vis2tex_lin(decode_vis2tex)

        num_p = [len(p) for p in all_proposals]

        rel_regs = vis2tex_tokens.split(num_p)

        tokens = []
        for reg,s,o in zip(rel_regs, ss,oo):

            sub = self.obj_semantics[torch.tensor(s).to(rel_pair_idx.device).long()].view(-1,768)
            obj = self.obj_semantics[torch.tensor(o).to(rel_pair_idx.device).long()].view(-1,768)

            MASK_TOKEN = self.mask_embedding(torch.tensor([0]).long().to(roi_features.device)) # 297 means @ special token.
            # MASK_TOKEN = self.text_transformer.token_embedding(torch.tensor([287]).long().to(roi_features.device)) # 287 means @ special token.

            if reg.shape[0] < 9:
                num_special_tokens = 9 - reg.shape[0]
                zero_tokens = [0 for _ in range(num_special_tokens)]
                vec = torch.tensor(zero_tokens).long().to(roi_features.device)
                paddings = self.text_transformer.token_embedding(vec)
                soft_prompt = torch.cat([reg, paddings, sub,  MASK_TOKEN, obj])
                tokens.append(soft_prompt)
            else:
                soft_prompt = torch.cat([reg,  sub,  MASK_TOKEN, obj])
                tokens.append(soft_prompt)

        tokens = torch.stack(tokens)

        mask_token_outputs = self.text_transformer(tokens, is_vis=True)
        return mask_token_outputs
        
def build_pretrain_head(cfg, in_channels):
    """
    Constructs a new relation head.
    By default, uses ROIRelationHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return PreTrainHead(cfg, in_channels)
