import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_temporal_pooling.modules.roi_temporal_pool import _RoITemporalPooling
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss
from model.utils.non_local_dot_product import NONLocalBlock3D

DEBUG = False


# Fewshot 3D CNN
class _TDCNN_Fewshot(nn.Module):
    """ faster RCNN """

    def __init__(self):
        super(_TDCNN_Fewshot, self).__init__()
        self.n_classes = cfg.NUM_CLASSES
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_twin = 0

        # define rpn
        # region proposal network
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.pooled_feat = None
        print('cfg.POOLING_LENGTH: %d' % cfg.POOLING_LENGTH)
        # self.RCNN_roi_temporal_pool = _RoITemporalPooling(cfg.POOLING_LENGTH, cfg.POOLING_HEIGHT, cfg.POOLING_WIDTH,
        #                                                   cfg.DEDUP_TWINS)
        self.RCNN_roi_temporal_pool = _RoITemporalPooling(1, 4, 4,
                                                          cfg.DEDUP_TWINS)
        if cfg.USE_ATTENTION:
            self.RCNN_attention = NONLocalBlock3D(self.dout_base_model, inter_channels=self.dout_base_model)

    def prepare_data(self, video_data):
        return video_data

    def extract_support_set_features(self, support_set_dataloader):
        all_features = []
        labels = []
        for step, (video_data, gt_twins, num_gt) in enumerate(support_set_dataloader):
            video_data = video_data.cuda()
            gt_twins = gt_twins.cuda().data
            batch_size = video_data.size(0)
            video_data = self.prepare_data(video_data)
            base_feat = self.RCNN_base(video_data)

            # Video length is always resized to 768
            # TODO: replicate the vectors here
            # TODO: verify whether this is correct
            rois = Variable(torch.Tensor([[0.0, 0.0, 767.0], [1.0, 0.0, 767.0]]).cuda(), requires_grad=False)
            if cfg.POOLING_MODE == 'pool':
                pooled_feat = self.RCNN_roi_temporal_pool(base_feat, rois.view(-1, 3))
            if cfg.USE_ATTENTION:
                pooled_feat = self.RCNN_attention(pooled_feat)
            fewshot_features = self._head_to_tail(pooled_feat)
            # Dimensions of fewshot_features: batch_size x 4096
            all_features.append(fewshot_features)
            labels.append(gt_twins[:, 0, 2])
        return all_features

    def forward(self, video_data, support_set_dataloader, gt_twins, whole_vid_for_testing=False):
        """
        :param video_data:
        :param support_set_dataloader: trimmed support set videos dataloader
        :param gt_twins: Ground truth timestamps + class label. Format: (start, end, label)
        :return:
        """
        support_set_features = self.extract_support_set_features(support_set_dataloader)

        batch_size = video_data.size(0)

        gt_twins = gt_twins.data
        # prepare data
        video_data = self.prepare_data(video_data)
        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(video_data)
        # print('base_feat dimensions: %s' % str(base_feat.shape))
        # feed base feature map tp RPN to obtain rois
        # rois, [rois_score], rpn_cls_prob, rpn_twin_pred, self.rpn_loss_cls, self.rpn_loss_twin, self.rpn_label, self.rpn_loss_mask
        rois, _, _, rpn_loss_cls, rpn_loss_twin, _, _ = self.RCNN_rpn(base_feat, gt_twins)

        # if it is training phase, then use ground truth twins for refining
        if self.training:
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = self.RCNN_proposal_target(rois, gt_twins)

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_twin = 0

            if whole_vid_for_testing:
                rois = torch.Tensor([[[0.0, 0.0, 767.0]]]).cuda()

        # print('rois shape: %s' % str(rois.shape))
        rois = Variable(rois)
        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'pool':

            # print('FINDING OUT DIFF BETWEEN GT_TWINS AND ROIS: %s, %s' % (str(gt_twins), str(rois.view(-1, 3))))
            reshaped_rois = rois.view(-1, 3)
            pooled_feat = self.RCNN_roi_temporal_pool(base_feat, reshaped_rois)
            # dimensions: 128 x 512 x 1 x 4 x 4
            # print('pooled_feat dim: %s' % str(pooled_feat.shape))

        if cfg.USE_ATTENTION:
            pooled_feat = self.RCNN_attention(pooled_feat)
            # feed pooled features to top model
        # NOTE!! Compare distances using this one
        pooled_feat = self._head_to_tail(pooled_feat)
        self.pooled_feat = pooled_feat
        # print('pooled_feat after head_to_tail dim: %s' % str(pooled_feat.shape))
        # compute twin offset, twin_pred will be (128, 42)
        twin_pred = self.RCNN_twin_pred(pooled_feat)
        # print('twin_pred shape: %s' % str(twin_pred.shape))

        if self.training:
            # select the corresponding columns according to roi labels, twin_pred will be (128, 2)
            twin_pred_view = twin_pred.view(twin_pred.size(0), int(twin_pred.size(1) / 2), 2)
            # print('twin_pred_view shape: %s' % str(twin_pred_view.shape))
            twin_pred_select = torch.gather(twin_pred_view, 1,
                                            rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 2))
            # print('twin_pred_select: %s' % str(twin_pred_select.shape))
            twin_pred = twin_pred_select.squeeze(1)
            # print('twin_pred: %s' % str(twin_pred.shape))

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, dim=1)

        if DEBUG:
            print("tdcnn.py--base_feat.shape {}".format(base_feat.shape))
            print("tdcnn.py--rois.shape {}".format(rois.shape))
            print("tdcnn.py--tdcnn_tail.shape {}".format(pooled_feat.shape))
            print("tdcnn.py--cls_score.shape {}".format(cls_score.shape))
            print("tdcnn.py--twin_pred.shape {}".format(twin_pred.shape))

        RCNN_loss_cls = 0
        RCNN_loss_twin = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_twin = _smooth_l1_loss(twin_pred, rois_target, rois_inside_ws, rois_outside_ws)

            # RuntimeError caused by mGPUs and higher pytorch version: https://github.com/jwyang/faster-rcnn.pytorch/issues/226
            rpn_loss_cls = torch.unsqueeze(rpn_loss_cls, 0)
            rpn_loss_twin = torch.unsqueeze(rpn_loss_twin, 0)
            RCNN_loss_cls = torch.unsqueeze(RCNN_loss_cls, 0)
            RCNN_loss_twin = torch.unsqueeze(RCNN_loss_twin, 0)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        twin_pred = twin_pred.view(batch_size, rois.size(1), -1)

        if self.training:
            return rois, cls_prob, twin_pred, rpn_loss_cls, rpn_loss_twin, RCNN_loss_cls, RCNN_loss_twin, rois_label
        else:
            return rois, cls_prob, twin_pred

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        self.RCNN_rpn.init_weights()
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_twin_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
