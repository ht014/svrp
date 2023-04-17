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
from .roi_relation_predictors import make_roi_relation_predictor
from .inference import make_roi_relation_post_processor
from .loss import make_roi_relation_loss_evaluator
from .sampling import make_roi_relation_samp_processor
from .pretrain_transformer import TransformerContext
from maskrcnn_benchmark.clip import clip
from .text_transformer import TextTransformer
from copy import deepcopy


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


class PreTrainHead(torch.nn.Module):
    

    def __init__(self, cfg, in_channels):
        super(PreTrainHead, self).__init__()

        self.cfg = cfg.clone()
        # same structure with box head, but different parameters
        # these param will be trained in a slow learning rate, while the parameters of box head will be fixed
        # Note: there is another such extractor in uniton_feature_extractor
        self.union_feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)

        self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        feat_dim = self.box_feature_extractor.out_channels
        self.predictor = make_roi_relation_predictor(cfg, feat_dim)
        self.post_processor = make_roi_relation_post_processor(cfg)
        self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)
        self.samp_processor = make_roi_relation_samp_processor(cfg)
        self.context_layer = TransformerContext(cfg, in_channels)

        # parameters
        self.text_transformer = TextTransformer(context_length=12)
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    # def load_text_encoder(self):
    #     model, _ = clip.load("/home/hetao/CLIP-main/img/RN101.pt",  device='cpu')
    #     self.text_transformer = model.transformer
    #     self.vocab_size = model.vocab_size
    #     self.token_embedding = model.token_embedding
    #     self.positional_embedding = model.positional_embedding
    #     self.ln_final = model.ln_final
    #     self.text_projection = model.text_projection
    #     self.logit_scale = model.logit_scale

    # def encode_text(self,text):
    #     x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
    #
    #     x = x + self.positional_embedding
    #     x = x.permute(1, 0, 2)  # NLD -> LND
    #     x = self.text_transformer(x)
    #     x = x.permute(1, 0, 2)  # LND -> NLD
    #     x = self.ln_final(x)
    #
    #     # x.shape = [batch_size, n_ctx, transformer.width]
    #     # take features from the eot embedding (eot_token is the highest number in each sequence)
    #     x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
    #     return x


    def sample_union_proposals(self, proposals, per_image_rois, context_length= 7,  max_unions = 5):
        sampled_proposals = []
        indexs = [i for i in range(len(proposals))]
        resort_rois = []
        sorted_proposals = []
        while len(sampled_proposals) < max_unions:
            sub = random.choice(indexs)
            obj = random.choice(indexs)
            if sub == obj: continue
            if sub > obj:
                t = sub
                sub = obj
                obj = t
            union_region_proposals = [] #[sub, obj]
            subject_box = proposals[torch.tensor([sub]).long().to(proposals.bbox.device)]
            object_box =  proposals[torch.tensor([obj]).long().to(proposals.bbox.device)]
            union_box =  boxlist_union(subject_box, object_box).bbox
            intersections = bbox_overlap(union_box,proposals.bbox)

            for i in range(intersections.shape[-1]):
                if intersections[0][i] < 0.95: continue
                union_region_proposals.append(i)
                if len(union_region_proposals) > context_length:
                    break

            if sub not in union_region_proposals:
                union_region_proposals.append(sub)
            if obj not in union_region_proposals:
                union_region_proposals.append(obj)
            # print("union_region_proposals length:",len(union_region_proposals))
            if union_region_proposals is not None:
                resort_rois.append(per_image_rois[union_region_proposals])
                boxes = proposals[torch.tensor(union_region_proposals).long().to(proposals.bbox.device)]
                sorted_proposals.append(boxes)
                sampled_proposals.append(union_region_proposals)

        return sampled_proposals,torch.cat(resort_rois,0),sorted_proposals


    def forward(self, features, proposals, targets=None, logger=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes. Note: it has been post-processed (regression, nms) in sgdet mode
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        roi_features = self.box_feature_extractor(features, proposals)
        num_objs = [len(p) for p in proposals]
        per_image_rois = roi_features.split(num_objs, dim=0)
        all_proposals = []
        all_rois = []
        for p,roi in zip(proposals,per_image_rois):
            _, sorted_rois, sorted_proposals = self.sample_union_proposals(p,roi)
            all_proposals += sorted_proposals
            all_rois.append(sorted_rois)
        roi_features = torch.cat(all_rois)
        image_features = self.context_layer(roi_features, all_proposals)

        z

        text_feats = [self.text_transformer(proposal.get_field('tokens').long()) for proposal in all_proposals]
        text_features = torch.cat(text_feats,0)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        ground_truth = torch.arange(logits_per_image.shape[0], dtype=torch.long, device=roi_features.device)
        img_loss = self.loss_img(logits_per_image, ground_truth)

        txt_lost = self.loss_txt(logits_per_text, ground_truth)

        output_losses = {'img_loss': img_loss, 'txt_loss': txt_lost}

        return roi_features, proposals, output_losses


def build_pretrain_head(cfg, in_channels):
    """
    Constructs a new relation head.
    By default, uses ROIRelationHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return PreTrainHead(cfg, in_channels)
