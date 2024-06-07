import numpy as np
import mindspore.nn as nn
from mindspore import context
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F

from src.ssh import SSH
from src.config import config
from src.proposal_generator import Proposal
from src.anchor_generator import AnchorGenerator
from src.bbox_assign_sample import BboxAssignSample


class SSHModel(nn.Cell):

    def __init__(self, config):
        super().__init__()
        self.dtype = np.float32
        self.ms_type = mstype.float32

        self.batch_size = config.batch_size
        self.num_layers = 3
        self.loss_reg_weight = Tensor(np.array(config.loss_reg_weight).astype(self.dtype))
        self.loss_cls_weight = Tensor(np.array(config.loss_cls_weight).astype(self.dtype))

        self.slice_index = ()
        self.feature_anchor_shape = ()
        self.slice_index += (0,)
        index = 0
        for shape in config.feature_shapes:
            self.slice_index += (self.slice_index[index] + shape[0] * shape[1] * config.num_anchors,)
            self.feature_anchor_shape += (shape[0] * shape[1] * config.num_anchors * config.batch_size,)
            index += 1
        self.num_expected_total = Tensor(np.array(config.num_expected_neg * self.batch_size).astype(self.dtype))

        self.anchor_scales = config.anchor_scales
        self.anchor_ratios = config.anchor_ratios
        self.anchor_base_sizes = config.anchor_base_sizes

        # anchor generator
        self.anchor_generators = []
        for i in range(len(self.anchor_base_sizes)):
            self.anchor_generators.append(
                AnchorGenerator(self.anchor_base_sizes[i], self.anchor_scales[i], self.anchor_ratios)
            )
        self.featmap_sizes = [(config.img_height // 8, config.img_width // 8),
                              (config.img_height // 16, config.img_width // 16),
                              (config.img_height // 32, config.img_width // 32)]
        self.anchor_list = self.get_anchors(self.featmap_sizes)
        self.proposal_generator = Proposal(config, 1, 2, True)
        self.proposal_generator.set_train_local(config, False)

        # net
        self.ssh = SSH()

        # box encoder
        self.num_bboxes = config.num_bboxes
        self.get_targets = BboxAssignSample(config, self.batch_size, self.num_bboxes, False)

        # operator
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.check_valid = P.CheckValid()
        self.squeeze = P.Squeeze()
        self.concat = P.Concat(axis=0)
        self.tile = P.Tile()
        self.loss_cls = P.SigmoidCrossEntropyWithLogits()
        self.loss_bbox = P.SmoothL1Loss(beta=1.0/9.0)
        self.sum_loss = P.ReduceSum()

        # content
        self.trans_shape = (0, 2, 3, 1)
        self.reshape_shape_reg = (-1, 4)
        self.reshape_shape_cls = (-1,)
        self.loss = Tensor(np.zeros((1,)).astype(self.dtype))
        self.cls_loss = Tensor(np.zeros((1,)).astype(self.dtype))
        self.reg_loss = Tensor(np.zeros((1,)).astype(self.dtype))


    def get_anchors(self, featmap_sizes):
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = ()
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_base_sizes[i])
            multi_level_anchors += (Tensor(anchors.astype(self.dtype)),)

        return multi_level_anchors

    def construct(self, inputs, img_metas, gt_bboxes, gt_labels, gt_valids):
        detect_out = self.ssh(inputs)
        if img_metas is not None:
            gt_bboxes = self.cast(gt_bboxes, mstype.float32)
            gt_labels = self.cast(gt_labels, mstype.float32)
            gt_valids = self.cast(gt_valids, mstype.float32)

        rpn_cls_score = ()
        rpn_bbox_pred = ()
        cls_score_total = ()
        bbox_pred_total = ()
        for i in range(3):
            x1, x2 = detect_out[i]
            cls_score_total = cls_score_total + (x1,)
            bbox_pred_total = bbox_pred_total + (x2,)

            x1 = self.transpose(x1, self.trans_shape)
            x1 = self.reshape(x1, self.reshape_shape_cls)
            x2 = self.transpose(x2, self.trans_shape)
            x2 = self.reshape(x2, self.reshape_shape_reg)
            rpn_cls_score = rpn_cls_score + (x1,)
            rpn_bbox_pred = rpn_bbox_pred + (x2,)

        loss = self.loss
        cls_loss = self.cls_loss
        reg_loss = self.reg_loss
        bbox_targets = ()
        bbox_weights = ()
        labels = ()
        label_weights = ()

        output = ()
        if self.training:
            for i in range(self.batch_size):
                multi_level_flags = ()
                anchor_list_tuple = ()

                for j in range(self.num_layers):
                    res = self.cast(self.check_valid(self.anchor_list[j], self.squeeze(img_metas[i:i + 1:1, ::])),
                                    mstype.int32)
                    multi_level_flags = multi_level_flags + (res,)
                    anchor_list_tuple = anchor_list_tuple + (self.anchor_list[j],)

                valid_flag_list = self.concat(multi_level_flags)
                anchor_using_list = self.concat(anchor_list_tuple)

                gt_bboxes_i = self.squeeze(gt_bboxes[i:i + 1:1, ::])
                gt_labels_i = self.squeeze(gt_labels[i:i + 1:1, ::])
                gt_valids_i = self.squeeze(gt_valids[i:i + 1:1, ::])

                bbox_target, bbox_weight, label, label_weight = self.get_targets(gt_bboxes_i,
                                                                                 gt_labels_i,
                                                                                 self.cast(valid_flag_list,
                                                                                           mstype.bool_),
                                                                                 anchor_using_list, gt_valids_i)

                bbox_target = self.cast(bbox_target, self.ms_type)
                bbox_weight = self.cast(bbox_weight, self.ms_type)
                label = self.cast(label, self.ms_type)
                label_weight = self.cast(label_weight, self.ms_type)

                for j in range(self.num_layers):
                    begin = self.slice_index[j]
                    end = self.slice_index[j + 1]
                    stride = 1
                    bbox_targets += (bbox_target[begin:end:stride, ::],)
                    bbox_weights += (bbox_weight[begin:end:stride],)
                    labels += (label[begin:end:stride],)
                    label_weights += (label_weight[begin:end:stride],)

            for i in range(3):
                bbox_target_using = ()
                bbox_weight_using = ()
                label_using = ()
                label_weight_using = ()

                for j in range(self.batch_size):
                    bbox_target_using += (bbox_targets[i + (self.num_layers * j)],)
                    bbox_weight_using += (bbox_weights[i + (self.num_layers * j)],)
                    label_using += (labels[i + (self.num_layers * j)],)
                    label_weight_using += (label_weights[i + (self.num_layers * j)],)

                bbox_target_with_batchsize = self.concat(bbox_target_using)
                bbox_weight_with_batchsize = self.concat(bbox_weight_using)
                label_with_batchsize = self.concat(label_using)
                label_weight_with_batchsize = self.concat(label_weight_using)

                # stop
                bbox_target_ = F.stop_gradient(bbox_target_with_batchsize)
                bbox_weight_ = F.stop_gradient(bbox_weight_with_batchsize)
                label_ = F.stop_gradient(label_with_batchsize)
                label_weight_ = F.stop_gradient(label_weight_with_batchsize)

                cls_score_i = self.cast(rpn_cls_score[i], self.ms_type)
                reg_score_i = self.cast(rpn_bbox_pred[i], self.ms_type)

                loss_cls = self.loss_cls(cls_score_i, label_)
                loss_cls_item = loss_cls * label_weight_
                loss_cls_item = self.sum_loss(loss_cls_item, (0,)) / self.num_expected_total

                loss_reg = self.loss_bbox(reg_score_i, bbox_target_)
                bbox_weight_ = self.tile(self.reshape(bbox_weight_, (self.feature_anchor_shape[i], 1)), (1, 4))
                loss_reg = loss_reg * bbox_weight_
                loss_reg_item = self.sum_loss(loss_reg, (1,))
                loss_reg_item = self.sum_loss(loss_reg_item, (0,)) / self.num_expected_total

                loss_total = self.loss_cls_weight * loss_cls_item + self.loss_reg_weight * loss_reg_item

                loss += loss_total
                cls_loss += loss_cls_item
                reg_loss += loss_reg_item

                output = (cls_loss, reg_loss)
        else:
            proposal, proposal_mask = self.proposal_generator(cls_score_total, bbox_pred_total, self.anchor_list)
            output = (proposal, proposal_mask)
        return output
