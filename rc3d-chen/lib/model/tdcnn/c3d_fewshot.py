import torch
import torch.nn as nn

from model.tdcnn.c3d import C3D
from model.tdcnn.tdcnn_fewshot import _TDCNN_Fewshot

class c3d_tdcnn_fewshot(_TDCNN_Fewshot):
    def __init__(self, pretrained=False):
        self.model_path = 'data/pretrained_model/activitynet_iter_30000_3fps-caffe.pth' #ucf101-caffe.pth' #c3d_sports1M.pth' #activitynet_iter_30000_3fps-caffe.pth
        self.dout_base_model = 512
        self.pretrained = pretrained
        _TDCNN_Fewshot.__init__(self)

    def _init_modules(self):
        c3d = C3D()
        if self.pretrained:
            print("Loading pretrained weights from %s" %(self.model_path))
            state_dict = torch.load(self.model_path)
            c3d.load_state_dict({k:v for k,v in state_dict.items() if k in c3d.state_dict()})

        # Using conv1 -> conv5b, not using the last maxpool
        self.RCNN_base = nn.Sequential(*list(c3d.features._modules.values())[:-1])
        # Using fc6
        self.RCNN_top = nn.Sequential(*list(c3d.classifier._modules.values())[:-4])
        # Fix the layers before pool2:
        for layer in range(6):
            for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

        # not using the last maxpool layer
        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)
        self.RCNN_twin_pred = nn.Linear(4096, 2 * self.n_classes)

    def _head_to_tail(self, pool5):
        pool5_flat = pool5.view(pool5.size(0), -1)
        fc6 = self.RCNN_top(pool5_flat)

        return fc6
