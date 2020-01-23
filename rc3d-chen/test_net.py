# --------------------------------------------------------
# Pytorch R-C3D
# Licensed under The MIT License [see LICENSE for details]
# Written by Shiguang Wang, based on code from Huijuan Xu
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import time
import cv2
import pickle
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import pickle
import random
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.twin_transform import clip_twins
from model.nms.nms_wrapper import nms
from model.rpn.twin_transform import twin_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.tdcnn.c3d import C3D
from model.tdcnn.c3d_fewshot import C3D, c3d_tdcnn_fewshot
from model.tdcnn.i3d import I3D, i3d_tdcnn
from model.utils.blob import prep_im_for_blob, video_list_to_blob
from model.tdcnn.resnet import resnet34, resnet50, resnet_tdcnn
from os.path import exists
from trainval_fewshot_net import create_sampled_support_set_dataset

# np.set_printoptions(threshold='nan')
DEBUG = False

xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a R-C3D network')
    parser.add_argument('--dataset', dest='dataset', default='thumos14', type=str,
                        help='test dataset')
    parser.add_argument('--net', dest='net', default='c3d', type=str, choices=['c3d', 'res18', 'res34', 'res50', 'eco'],
                        help='main network c3d, i3d, res34, res50')
    parser.add_argument('--set', dest='set_cfgs', nargs=argparse.REMAINDER,
                        help='set config keys', default=None)
    parser.add_argument('--load_dir', dest='load_dir', type=str,
                        help='directory to load models', default="./models")
    parser.add_argument('--output_dir', dest='output_dir', type=str,
                        help='directory for the log files', default="./output")
    parser.add_argument('--cuda', dest='cuda', action='store_true',
                        help='whether use CUDA')
    parser.add_argument('--checksession', default=1, type=int,
                        help='checksession to load model')
    parser.add_argument('--checkepoch', default=1, type=int,
                        help='checkepoch to load network')
    parser.add_argument('--checkpoint', default=9388, type=int,
                        help='checkpoint to load network')
    parser.add_argument('--nw', dest='num_workers', default=8, type=int,
                        help='number of worker to load data')
    parser.add_argument('--bs', dest='batch_size', default=1, type=int,
                        help='batch_size, only support batch_size=1')
    parser.add_argument('--vis', dest='vis', action='store_true',
                        help='visualization mode')
    parser.add_argument('--roidb_dir', dest='roidb_dir', default="./preprocess",
                        help='roidb_dir')
    parser.add_argument('--gpus', dest='gpus', nargs='+', type=int, default=0,
                        help='gpu ids.')
    parser.add_argument('--softmax_fewshot_score', dest='softmax_fewshot_score', action='store_true')
    parser.add_argument('--fewshot_score_threshold', dest='fewshot_score_threshold', type=float)
    args = parser.parse_args()
    return args


def get_roidb(path):
    data = pickle.load(open(path, 'rb'))
    return data


combined_pooled_feat = []


def comp_pairwise_dist():
    num_vecs = len(combined_pooled_feat)
    dist = np.zeros([num_vecs, num_vecs])
    for i in range(num_vecs):
        for j in range(num_vecs):
            dist[i][j] = torch.norm(combined_pooled_feat[i] - combined_pooled_feat[j])
        sorted_vals, sorted_idx = torch.sort(torch.Tensor(dist[i]))
        print('Those closest to %d are %s' % (i, str(sorted_idx[:5])))


def test_net(tdcnn_demo, dataloader, args, trimmed_support_set_roidb, thresh=0.7, use_softmax_fewshot_score=False):
    FEWSHOT_FEATURES_PATH = '/home/vltava/fewshot_features_5_shot.pkl'

    start = time.time()

    _t = {'im_detect': time.time(), 'misc': time.time()}

    tdcnn_demo.eval()

    if exists(FEWSHOT_FEATURES_PATH):
        all_fewshot_features = pickle.load(open(FEWSHOT_FEATURES_PATH, 'rb'))
    else:
        support_set_dataloader = create_sampled_support_set_dataset(trimmed_support_set_roidb,
                                                                    [2, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19],
                                                                    batch_size=args.batch_size,
                                                                    num_workers=args.num_workers,
                                                                    samples_per_class=5)
        print('Loading fewshot trimmed videos')
        feature_vectors = []
        feature_labels = []
        for _, (video_data, gt_twins, _) in enumerate(support_set_dataloader):
            support_set_features, support_set_labels = \
                tdcnn_demo(video_data, None, None, gt_twins, True)
            feature_vectors.append(support_set_features)
            feature_labels.append(support_set_labels)
            del video_data
        all_fewshot_features = (torch.cat(feature_vectors), torch.cat(feature_labels))

        pickle.dump(all_fewshot_features, open(FEWSHOT_FEATURES_PATH, 'wb'))

    support_set_features = all_fewshot_features[0]
    support_set_labels = all_fewshot_features[1]

    # print(f'Got {support_set_features.shape[0]} few shot features')

    unique_support_set_labels = support_set_labels.cpu().unique(sorted=True).cuda()

    data_tic = time.time()
    for i, (video_data, gt_twins, num_gt, video_info, fewshot_label) in enumerate(dataloader):
        video_data = video_data.cuda()
        gt_twins = gt_twins.cuda()
        batch_size = video_data.shape[0]
        data_toc = time.time()
        data_time = data_toc - data_tic

        batch_support_set_size = 5

        unique_labels_in_test = fewshot_label.cpu().unique().numpy().tolist()
        batch_support_set_labels = unique_labels_in_test
        if len(unique_labels_in_test) < batch_support_set_size:
            other_labels = torch.cat([l.unsqueeze(0) for l in unique_support_set_labels if l not in unique_labels_in_test]).cpu().numpy().tolist()
            batch_support_set_labels = unique_labels_in_test + random.sample(other_labels, batch_support_set_size - len(unique_labels_in_test))
        batch_support_set_indices = [i for i, v in enumerate(support_set_labels) if v in batch_support_set_labels]
        batch_support_set_features = support_set_features[batch_support_set_indices]
        batch_support_set_labels = support_set_labels[batch_support_set_indices]
        unique_batch_support_set_labels = batch_support_set_labels.cpu().unique(sorted=True).cuda()

        det_tic = time.time()
        rois, cls_prob, twin_pred, fewshot_scores, fewshot_scores_softmax = \
            tdcnn_demo(video_data,
                       torch.cat(args.batch_size * [batch_support_set_features.unsqueeze(0)]),
                       torch.cat(args.batch_size * [batch_support_set_labels.unsqueeze(0)]),
                       gt_twins)

        scores_all = fewshot_scores.data
        twins = rois.data[:, :, 1:3]

        if cfg.TEST.TWIN_REG:
            # Apply bounding-twin regression deltas
            twin_deltas = twin_pred.data
            if cfg.TRAIN.TWIN_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                twin_deltas = twin_deltas.view(-1, 2) * torch.FloatTensor(cfg.TRAIN.TWIN_NORMALIZE_STDS).type_as(
                    twin_deltas) \
                              + torch.FloatTensor(cfg.TRAIN.TWIN_NORMALIZE_MEANS).type_as(twin_deltas)
                twin_deltas = twin_deltas.view(batch_size, -1, 2 * args.num_classes)

            pred_twins_all = twin_transform_inv(twins, twin_deltas, batch_size)
            pred_twins_all = clip_twins(pred_twins_all, cfg.TRAIN.LENGTH[0], batch_size)
        else:
            # Simply repeat the twins, once for each class
            pred_twins_all = np.tile(twins, (1, scores_all.shape[1]))

        det_toc = time.time()
        detect_time = det_toc - det_tic

        for b in range(batch_size):
            misc_tic = time.time()
            print(video_info[b])
            # cls_prob scores are not helpful for fewshot since these are 21-class output (0 = background)
            # and the new example doesn't appear in the training data
            pred_twins = pred_twins_all[b]  # .squeeze()

            fewshot_scores_of_batch = torch.zeros(fewshot_scores.shape[1], batch_support_set_size).cuda()
            fewshot_scores_softmax_of_batch = torch.zeros(fewshot_scores.shape[1], batch_support_set_size).cuda()

            has_detections = False

            for j in range(batch_support_set_size):
                idx_of_label_with_id = (batch_support_set_labels == unique_batch_support_set_labels[j]).nonzero().squeeze()
                fewshot_scores_of_batch[:, j] = fewshot_scores[b, :, idx_of_label_with_id].mean(dim=1)  # Average scores of the same label
                fewshot_scores_softmax_of_batch[:, j] = fewshot_scores_softmax[b, :, idx_of_label_with_id].sum(dim=1)

                if use_softmax_fewshot_score:
                    inds = torch.nonzero(fewshot_scores_softmax_of_batch[:, j] > thresh).view(-1)
                else:
                    inds = torch.nonzero(fewshot_scores_of_batch[:, j] > thresh).view(-1)

                label_id = unique_batch_support_set_labels[j].item()

                # if there is detection
                if inds.numel() > 0:
                    has_detections = True
                    cls_scores = fewshot_scores_of_batch[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    # This doesn't quite make sense because label_id column has no meaning to the network
                    cls_twins = pred_twins[inds][:, label_id * 2:(label_id + 1) * 2]

                    cls_dets = torch.cat((cls_twins, cls_scores.unsqueeze(1)), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_dets, cfg.TEST.NMS)
                    if len(keep) > 0:
                        cls_dets = cls_dets[keep.view(-1).long()]
                        print("activity: ", label_id)
                        print(cls_dets.cpu().numpy())
                else:
                    pass
                    # DEBUGGING ONLY. If this is the correct label but no detection, return scores
                    # if label_id == fewshot_label[b].item():
                        # print(f'**** FAILED TO DETECT CLASS: {fewshot_scores_softmax_of_batch[:, j].mean()}')

            most_likely_labels = unique_batch_support_set_labels[torch.sort(fewshot_scores_of_batch, descending=True)[1]][:50]
            most_likely_labels_softmax = unique_batch_support_set_labels[torch.sort(fewshot_scores_softmax_of_batch, descending=True)[1]][:50]

            if not has_detections:
                sorted_scores, sorted_scores_idx = torch.sort(fewshot_scores_softmax_of_batch, descending=True)
                sorted_scores = sorted_scores[:, 0]
                sorted_scores_idx = sorted_scores_idx[:, 0]
                sorted_scores_rows, sorted_scores_rows_idx = torch.sort(sorted_scores, descending=True)
                sorted_scores_rows = sorted_scores_rows[:10]
                sorted_scores_rows_idx = sorted_scores_rows_idx[:10]
                sorted_scores_cols_idx = sorted_scores_idx[sorted_scores_rows_idx]
                unique_cols_idx = sorted_scores_cols_idx.cpu().unique(sorted=True).cuda()
                for label_idx in unique_cols_idx:
                    cls_twins = pred_twins[sorted_scores_rows_idx][:, label_idx * 2:(label_idx + 1) * 2]
                    cls_dets = torch.cat((cls_twins, sorted_scores_rows.unsqueeze(1)), 1)
                # print(f'No detections. Most likely labels are {most_likely_labels}, {most_likely_labels_softmax}')

            misc_toc = time.time()
            nms_time = misc_toc - misc_tic
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s {:.3f}s' \
                  .format(i * batch_size + b + 1, args.num_videos, data_time / batch_size, detect_time / batch_size,
                          nms_time))

        if args.vis:
            pass

        data_tic = time.time()
    end = time.time()
    print("test time: %0.4fs" % (end - start))


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "thumos14":
        # args.imdb_name = "train_data_25fps_flipped.pkl"
        args.imdbval_name = "val_data_25fps.pkl"
        args.num_classes = 21
        args.set_cfgs = ['ANCHOR_SCALES', '[2,4,5,6,8,9,10,12,14,16]', 'NUM_CLASSES', args.num_classes]
        # args.set_cfgs = ['ANCHOR_SCALES', '[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56]', 'NUM_CLASSES', args.num_classes]
    elif args.dataset == "activitynet":
        # args.imdb_name = "train_data_5fps_flipped.pkl"
        args.imdbval_name = "val_data_25fps.pkl"
        args.num_classes = 201
        # args.set_cfgs = ['ANCHOR_SCALES', '[1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64]', 'NUM_CLASSES', args.num_classes]
        args.set_cfgs = ['ANCHOR_SCALES',
                         '[1,1.25, 1.5,1.75, 2,2.5, 3,3.5, 4,4.5, 5,5.5, 6,7, 8,9,10,11,12,14,16,18,20,22,24,28,32,36,40,44,52,60,68,76,84,92,100]',
                         'NUM_CLASSES', args.num_classes]

    args.cfg_file = "cfgs/{}_{}.yml".format(args.net, args.dataset)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda
    cfg.CUDA = args.cuda

    print('Using config:')
    pprint.pprint(cfg)

    trimmed_support_set_roidb_path = os.path.join(args.roidb_dir, args.dataset, "trimmed_14_cls.pkl")
    trimmed_support_set_roidb = get_roidb(trimmed_support_set_roidb_path)

    untrimmed_test_roidb_path = args.roidb_dir + "/" + args.dataset + "/" + args.imdbval_name
    untrimmed_test_roidb = get_roidb(untrimmed_test_roidb_path)
    untrimmed_test_dataset = roibatchLoader(untrimmed_test_roidb, phase='test')
    untrimmed_test_dataloader = torch.utils.data.DataLoader(untrimmed_test_dataset, batch_size=args.batch_size,
                                                            num_workers=args.num_workers, shuffle=True)

    num_videos = len(untrimmed_test_dataset)
    args.num_videos = num_videos
    print('{:d} roidb entries'.format(num_videos))

    model_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(model_dir):
        raise Exception('There is no input directory for loading network from ' + model_dir)
    output_dir = args.output_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    load_name = os.path.join(model_dir, 'tdcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    # load_name = '/home/vltava/few-shot-activity-localization/rc3d-chen/models/c3d/thumos14/without_metatraining_models/tdcnn_1_7_1960.pth'

    # initilize the network here.
    if args.net == 'c3d':
        tdcnn_demo = c3d_tdcnn_fewshot(pretrained=False)
    elif args.net == 'res18':
        tdcnn_demo = resnet_tdcnn(depth=18, pretrained=False)
    elif args.net == 'res34':
        tdcnn_demo = resnet_tdcnn(depth=34, pretrained=False)
    elif args.net == 'res50':
        tdcnn_demo = resnet_tdcnn(depth=50, pretrained=False)
    else:
        print("network is not defined")

    tdcnn_demo.create_architecture()
    # save memory
    for key, value in tdcnn_demo.named_parameters(): value.requires_grad = False
    print(tdcnn_demo)

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    tdcnn_demo.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
        print('load model successfully!')

    if args.cuda and torch.cuda.is_available():
        tdcnn_demo = tdcnn_demo.cuda()
        if isinstance(args.gpus, int):
            args.gpus = [args.gpus]
        # assert len(args.gpus) == args.batch_size, "only support one batch_size for one gpu"
        tdcnn_demo = nn.parallel.DataParallel(tdcnn_demo, device_ids=args.gpus)

    print(f'Using {"softmax " if args.softmax_fewshot_score else " "}threshold of {args.fewshot_score_threshold}')

    test_net(tdcnn_demo, untrimmed_test_dataloader, args, trimmed_support_set_roidb, args.fewshot_score_threshold,
             args.softmax_fewshot_score)
