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

SUPPORT_SET_SIZE = 4

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

    def forward_support_set(self, video_data, gt_twins):
        video_data = video_data.cuda()
        gt_twins = gt_twins.cuda().data
        batch_size = video_data.size(0)
        video_data = self.prepare_data(video_data)
        base_feat = self.RCNN_base(video_data)

        rois_tensor = torch.zeros(batch_size, 3)
        rois_tensor[:, 0] = torch.tensor([i for i in range(batch_size)], dtype=torch.float)
        # Video is always resized to have length 768
        rois_tensor[:, 2] = 767.0
        rois = Variable(rois_tensor.cuda(), requires_grad=False)
        if cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_temporal_pool(base_feat, rois.view(-1, 3))
        if cfg.USE_ATTENTION:
            pooled_feat = self.RCNN_attention(pooled_feat)
        fewshot_features = self._head_to_tail(pooled_feat)
        # Dimensions of fewshot_features: batch_size x 4096
        return fewshot_features, torch.tensor(gt_twins[:, 0, 2], dtype=torch.int64).cuda()

    def compute_fewshot_cls_loss(self, support_set_features, support_set_labels, test_features, rois_label):
        """
        Fewshot classification loss by comparing test video features against support set video features
        :param support_set_features: [SUPPORT_SET_SIZEx4096]
        :param support_set_labels: [SUPPORT_SET_SIZE]
        :param test_features: [128x4096]
        :param rois_label: [128]
        :return A tensor with a single number representing the cross entropy loss
        """
        valid_test_features_idx = [i for i in range(rois_label.shape[0]) if rois_label[i] in support_set_labels]
        valid_features = test_features[valid_test_features_idx]
        valid_labels = rois_label[valid_test_features_idx]
        similarities = torch.zeros(valid_features.shape[0], support_set_labels.shape[0]).cuda()
        for i in range(support_set_labels.shape[0]):
            similarities[:, i] = F.cosine_similarity(support_set_features[i].unsqueeze(0), valid_features)
        # softmax smoothed out things too much
        # softmax_out = F.softmax(similarities, dim=1).cuda()
        # TODO: this assumes that each class has only one example. Handle the case when there are multiple examples
        labels_idx = torch.tensor([(support_set_labels == j).nonzero().item() for j in valid_labels], dtype=torch.int64).cuda()
        cross_entropy = F.cross_entropy(similarities, labels_idx)

        max_idx = torch.argmax(similarities, dim=1)
        correct_count = len((max_idx == labels_idx).nonzero())
        print(f'Correct predictions: {correct_count} out of {len(labels_idx)}. Cross entropy loss: {cross_entropy}')

        return cross_entropy

    def compute_fewshot_predictions(self, support_set_features, support_set_labels, test_features):
        similarities = torch.zeros(test_features.shape[0], support_set_labels.shape[0]).cuda()
        for i in range(support_set_labels.shape[0]):
            similarities[:, i] = F.cosine_similarity(support_set_features[i].unsqueeze(0), test_features)
        softmaxed = F.softmax(similarities, dim=1).cuda()
        return similarities.unsqueeze(0), softmaxed.unsqueeze(0)
        # max_scores = torch.max(similarities, dim=1)
        # Note: this won't work if batch_size of this thread > 1. Need to handle batch input
        # return max_scores[0].unsqueeze(0), max_scores[1].unsqueeze(0)

    def forward(self, video_data, support_set_features, support_set_labels, gt_twins, is_support_set=False):
        """
        :param video_data:
        :param support_set_features: Trimmed support set video features
        :param support_set_labels: Support set feature labels
        :param gt_twins: Ground truth timestamps + class label. Format: (start, end, label)
        :return:
        """
        if is_support_set:
            return self.forward_support_set(video_data, gt_twins)

        # NOTE: this method is called on multiple threads. So batch_size could be 1 here even though it's set to 2
        # indicating that this is executed on multiple GPUs
        batch_size = video_data.size(0)

        support_set_features = support_set_features.squeeze()
        support_set_labels = support_set_labels.squeeze()
        # assert support_set_labels.shape[0] == SUPPORT_SET_SIZE
        # assert support_set_features.shape[0] == SUPPORT_SET_SIZE

        gt_twins = gt_twins.data
        # prepare data
        video_data = self.prepare_data(video_data)
        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(video_data)
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

        rois = Variable(rois)
        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'pool':

            reshaped_rois = rois.view(-1, 3)
            pooled_feat = self.RCNN_roi_temporal_pool(base_feat, reshaped_rois)
            # dimensions: 128 x 512 x 1 x 4 x 4

        if cfg.USE_ATTENTION:
            pooled_feat = self.RCNN_attention(pooled_feat)
            # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)
        # compute twin offset, twin_pred will be (128, 42)
        twin_pred = self.RCNN_twin_pred(pooled_feat)

        if self.training:
            # select the corresponding columns according to roi labels, twin_pred will be (128, 2)
            twin_pred_view = twin_pred.view(twin_pred.size(0), int(twin_pred.size(1) / 2), 2)
            twin_pred_select = torch.gather(twin_pred_view, 1,
                                            rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 2))
            twin_pred = twin_pred_select.squeeze(1)

            fewshot_cls_loss = self.compute_fewshot_cls_loss(support_set_features, support_set_labels, pooled_feat, rois_label)
        else:
            fewshot_pred, fewshot_pred_softmax = self.compute_fewshot_predictions(support_set_features, support_set_labels, pooled_feat)

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
            # Wrapping fewshot_cls_loss with array to fix "tensor has no dimensions" error when gathering outputs
            return rois, cls_prob, twin_pred, rpn_loss_cls, rpn_loss_twin, RCNN_loss_cls, RCNN_loss_twin, rois_label,\
                   fewshot_cls_loss.unsqueeze(0).cuda()
        else:
            return rois, cls_prob, twin_pred, fewshot_pred, fewshot_pred_softmax

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
