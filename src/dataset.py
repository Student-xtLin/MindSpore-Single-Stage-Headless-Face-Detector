import os
import cv2
import numpy as np
from numpy import random

import mindspore as ms
from mindspore import dataset
from mindspore.mindrecord import FileWriter


class Expand:
    """expand image"""

    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, img, boxes):
        if random.randint(2):
            return img, boxes

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        img = expand_img
        boxes += np.tile((left, top), 2)
        return img, boxes


def flip_column(img, img_shape, gt_bboxes):
    """flip operation for image"""
    img_data = img
    img_data = np.flip(img_data, axis=1)
    flipped = gt_bboxes.copy()
    _, w, _ = img_data.shape

    flipped[..., 0::4] = w - gt_bboxes[..., 2::4] - 1
    flipped[..., 2::4] = w - gt_bboxes[..., 0::4] - 1

    return (img_data, img_shape, flipped)


def rescale_with_tuple(img, scale):
    h, w = img.shape[:2]
    scale_factor = min(max(scale) / max(h, w), min(scale) / min(h, w))
    new_size = int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5)
    rescaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

    return rescaled_img, scale_factor


def rescale_column_test(img, img_shape, gt_bboxes, config):
    """rescale operation for image of eval"""
    img_data, scale_factor = rescale_with_tuple(img, (config.img_width, config.img_height))
    if img_data.shape[0] > config.img_height:
        img_data, scale_factor2 = rescale_with_tuple(img_data, (config.img_height, config.img_height))
        scale_factor = scale_factor * scale_factor2

    pad_h = config.img_height - img_data.shape[0]
    pad_w = config.img_width - img_data.shape[1]
    assert ((pad_h >= 0) and (pad_w >= 0))

    pad_img_data = np.zeros((config.img_height, config.img_width, 3)).astype(img_data.dtype)
    pad_img_data[0:img_data.shape[0], 0:img_data.shape[1], :] = img_data

    img_shape = np.append(img_shape, (scale_factor, scale_factor))
    img_shape = np.asarray(img_shape, dtype=np.float32)

    return (pad_img_data, img_shape, gt_bboxes)


def imnormalize_column(img, img_shape, gt_bboxes):
    """imnormalize operation for image"""
    mean = np.asarray([123.675, 116.28, 103.53])
    std = np.asarray([58.395, 57.12, 57.375])
    img_data = img.copy().astype(np.float32)
    cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB, img_data)  # inplace
    cv2.subtract(img_data, np.float64(mean.reshape(1, -1)), img_data)  # inplace
    cv2.multiply(img_data, 1 / np.float64(std.reshape(1, -1)), img_data)  # inplace

    img_data = img_data.astype(np.float32)
    return (img_data, img_shape, gt_bboxes)


def transpose_column(img, img_shape, gt_bboxes):
    """transpose operation for image"""
    img_data = img.transpose(2, 0, 1).copy()
    img_data = img_data.astype(np.float32)
    img_shape = img_shape.astype(np.float32)
    gt_bboxes = gt_bboxes.astype(np.float32)

    return (img_data, img_shape, gt_bboxes)


def rescale_column(img, img_shape, gt_bboxes, config):
    """rescale operation for image"""
    img_data, scale_factor = rescale_with_tuple(img, (config.img_width, config.img_height))
    if img_data.shape[0] > config.img_height:
        img_data, scale_factor2 = rescale_with_tuple(img_data, (config.img_height, config.img_height))
        scale_factor = scale_factor * scale_factor2

    gt_bboxes = gt_bboxes * scale_factor
    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_data.shape[1] - 1)
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_data.shape[0] - 1)

    pad_h = config.img_height - img_data.shape[0]
    pad_w = config.img_width - img_data.shape[1]
        
    assert ((pad_h >= 0) and (pad_w >= 0))

    pad_img_data = np.zeros((config.img_height, config.img_width, 3)).astype(img_data.dtype)
    pad_img_data[0:img_data.shape[0], 0:img_data.shape[1], :] = img_data

    img_shape = (config.img_height, config.img_width, 1.0)
    img_shape = np.asarray(img_shape, dtype=np.float32)

    return (pad_img_data, img_shape, gt_bboxes)


def resize_column(img, img_shape, gt_bboxes, config):
    """resize operation for image"""
    img_data = img
    h, w = img_data.shape[:2]
    img_data = cv2.resize(
        img_data, (config.img_width, config.img_height), interpolation=cv2.INTER_LINEAR)
    h_scale = config.img_height / h
    w_scale = config.img_width / w

    scale_factor = np.array(
        [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
    img_shape = (config.img_height, config.img_width, 1.0)
    img_shape = np.asarray(img_shape, dtype=np.float32)

    gt_bboxes = gt_bboxes * scale_factor

    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)

    return (img_data, img_shape, gt_bboxes)


def resize_column_test(img, img_shape, gt_bboxes, config):
    """resize operation for image of eval"""
    img_data = img
    h, w = img_data.shape[:2]
    img_data = cv2.resize(
        img_data, (config.img_width, config.img_height), interpolation=cv2.INTER_LINEAR)
    h_scale = config.img_height / h
    w_scale = config.img_width / w

    scale_factor = np.array(
        [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
    img_shape = np.append(img_shape, (h_scale, w_scale))
    img_shape = np.asarray(img_shape, dtype=np.float32)

    gt_bboxes = gt_bboxes * scale_factor

    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)

    return (img_data, img_shape, gt_bboxes)


def expand_column(img, img_shape, gt_bboxes):
    """expand operation for image"""
    expand = Expand()
    img, gt_bboxes = expand(img, gt_bboxes)

    return (img, img_shape, gt_bboxes)


def preprocess_fn(image, box, is_training, config):
    """Preprocess function for dataset."""

    def _infer_data(image_bgr, image_shape, gt_box_new, gt_label, gt_valid_new):
        image_shape = image_shape[:2]
        input_data = image_bgr, image_shape, gt_box_new

        if config.keep_ratio:
            input_data = rescale_column_test(*input_data, config=config)
        else:
            input_data = resize_column_test(*input_data, config=config)
        input_data = imnormalize_column(*input_data)

        output_data = transpose_column(*input_data)
        return *output_data, gt_label, gt_valid_new

    def _data_aug(image, box, is_training):
        """Data augmentation function."""
        pad_max_number = config.num_gts
        if pad_max_number < box.shape[0]:
            box = box[:pad_max_number, :]
        image_bgr = image.copy()
        image_bgr[:, :, 0] = image[:, :, 2]
        image_bgr[:, :, 1] = image[:, :, 1]
        image_bgr[:, :, 2] = image[:, :, 0]
        image_shape = image_bgr.shape[:2]
        gt_box, gt_valid = box[:, :4], box[:, 4]
        gt_label = np.ones(pad_max_number, dtype=np.uint8)
        gt_box_new = np.pad(gt_box, ((0, pad_max_number - box.shape[0]), (0, 0)), mode="constant", constant_values=0)
        gt_valid_new = np.pad(gt_valid, (0, pad_max_number - box.shape[0]), mode="constant", constant_values=0)

        if not is_training:
            return _infer_data(image_bgr, image_shape, gt_box_new, gt_label, gt_valid_new)

        flip = (np.random.rand() < config.flip_ratio)
        expand = (np.random.rand() < config.expand_ratio)
        input_data = image_bgr, image_shape, gt_box_new

        if expand:
            input_data = expand_column(*input_data)
        if config.keep_ratio:
            input_data = rescale_column(*input_data, config=config)
        else:
            input_data = resize_column(*input_data, config=config)
        input_data = imnormalize_column(*input_data)
        if flip:
            input_data = flip_column(*input_data)

        output_data = transpose_column(*input_data)
        return *output_data, gt_label, gt_valid_new

    return _data_aug(image, box, is_training)


def create_train_data_from_txt(image_dir, anno_path):
    image_files = []
    image_anno_dict = {}
    if not os.path.isdir(image_dir):
        raise RuntimeError("Path given is not valid.")
    if not os.path.isfile(anno_path):
        raise RuntimeError("Annotation file is not valid.")

    with open(anno_path, "rb") as f:
        lines = f.readlines()
    for i in range(len(lines)):
        file_str = lines[i].decode("utf-8").strip()
        if file_str.endswith('jpg'):
            image_path = os.path.join(image_dir, file_str)
            i += 1
            nums = int(lines[i])
            bboxs = []
            for j in range(nums):
                i += 1
                bbox_str = lines[i].decode("utf-8").strip()
                line_split = str(bbox_str).split(' ')
                if line_split[-3] == '0':
                    bboxs.append([int(line_split[0]), int(line_split[1]), int(line_split[0])+int(line_split[2]), \
                        int(line_split[1])+int(line_split[3]), 1])
                if len(bboxs) > 0 and j == nums-1:
                    image_files.append(image_path)
                    image_anno_dict[image_path] = bboxs
    return image_files, image_anno_dict


def data_to_mindrecord_byte_image(config, prefix="wider.mindrecord", file_num=8):
    """Create MindRecord file."""
    mindrecord_dir = config.mindrecord_dir
    mindrecord_path = os.path.join(mindrecord_dir, prefix)
    writer = FileWriter(mindrecord_path, file_num)

    image_files, image_anno_dict = create_train_data_from_txt(config.image_dir, config.anno_path)

    wider_json = {
        "image": {"type": "bytes"},
        "annotation": {"type": "int32", "shape": [-1, 5]},
    }
    writer.add_schema(wider_json, "wider_json")

    num = 100   # num=200?
    for image_name in image_files:
        if num == 0:
            break
        num -= 1
        with open(image_name, 'rb') as f:
            img = f.read()
        annos = np.array(image_anno_dict[image_name], dtype=np.int32)
        row = {"image": img, "annotation": annos}
        writer.write_raw_data([row])
    writer.commit()


def create_wider_dataset(config, mindrecord_file, batch_size=2, device_num=1, rank_id=0, is_training=True,
                              num_parallel_workers=8, python_multiprocessing=False):
    """Create Wider dataset with MindDataset."""
    cv2.setNumThreads(0)
    dataset.config.set_prefetch_size(8)
    ds = dataset.MindDataset(mindrecord_file, columns_list=["image", "annotation"], num_shards=device_num, shard_id=rank_id,
                        num_parallel_workers=4, shuffle=is_training)
    decode = ms.dataset.vision.Decode()
    ds = ds.map(input_columns=["image"], operations=decode)
    compose_map_func = (lambda image, annotation: preprocess_fn(image, annotation, is_training, config=config))

    if is_training:
        ds = ds.map(input_columns=["image", "annotation"],
                    output_columns=["image", "image_shape", "box", "label", "valid_num"],
                   # column_order=["image", "image_shape", "box", "label", "valid_num"],    # 由于API更改，删除
                    operations=compose_map_func, python_multiprocessing=python_multiprocessing,
                    num_parallel_workers=num_parallel_workers)
        ds = ds.project(["image", "image_shape", "box", "label", "valid_num"])              # 新增
        ds = ds.batch(batch_size, drop_remainder=True)
    else:
        ds = ds.map(input_columns=["image", "annotation"],
                    output_columns=["image", "image_shape", "box", "label", "valid_num"],
                   # column_order=["image", "image_shape", "box", "label", "valid_num"],    # 由于API更改，删除
                    operations=compose_map_func,
                    num_parallel_workers=num_parallel_workers)
        ds = ds.project(["image", "image_shape", "box", "label", "valid_num"])              # 新增
        ds = ds.batch(batch_size, drop_remainder=True)
    return ds
