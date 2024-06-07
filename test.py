import os
import cv2
import time
import numpy as np

from mindspore import ops, Tensor
from mindspore import context, load_checkpoint, load_param_into_net

from src.config import config
from src.ssh_model import SSHModel
from src.dataset import data_to_mindrecord_byte_image, create_wider_dataset

rank = 0
device_num = 1
context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target, device_id=0)


def pre_process(img):
    img = cv2.resize(img, (img.shape[1]//16*16, img.shape[0]//16*16))
    print('img.shape:',img.shape)   # (912, 1024, 3)    (1120, 1024, 3)
    mean = np.asarray([123.675, 116.28, 103.53])
    std = np.asarray([58.395, 57.12, 57.375])
    img_data = img.copy().astype(np.float32)
    cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB, img_data)  # inplace
    cv2.subtract(img_data, np.float64(mean.reshape(1, -1)), img_data)  # inplace
    cv2.multiply(img_data, 1 / np.float64(std.reshape(1, -1)), img_data)  # inplace

    img_data = img_data.astype(np.float32)
    img_data = np.transpose(img_data, (2, 0, 1))
    return np.expand_dims(img_data, axis=0)


def box_process(out, thr=0.9):
    nms = ops.NMSWithMask(0.01)
    box, ind, mask = nms(out[0][0])
    box, ind, mask = box.asnumpy(), ind.asnumpy(), mask.asnumpy()
    box = box[mask]
    mask_2 = []
    for i in range(len(box)):
        if box[i][-1] < thr:
            mask_2.append(False)
        else:
            mask_2.append(True)
    box = box[mask_2]
    return box


def post_process(img):
    img = np.uint8((img - np.min(img)) / (np.max(img) - np.min(img)) * 255)
    img = np.transpose(np.squeeze(img, axis=0), (1, 2, 0))
    img[:, :, 0], img[:, :, 2] = img[:, :, 2].copy(), img[:, :, 0].copy()
    return img


def infer_ssh():
    net = SSHModel(config)
    net = net.set_train(False)

    param_dict = load_checkpoint('./ckpts/ckpt_0/ssh.ckpt')
    load_param_into_net(net, param_dict)
    
    test_dir = './test/'
    # test_dir = './failed_img/'
    test_img = os.listdir(test_dir)
    output_dir = './test_result/'
    for i in test_img:
        print(i)
        img = cv2.imread(test_dir + i)
        img = pre_process(img)
        out = net(Tensor(img), None, None, None, None)
        box = box_process(out, 0.9)
        img = post_process(img)

        for j in range(len(box)):
            cv2.rectangle(img, (int(box[j][0]), int(box[j][1])), (int(box[j][2]), int(box[j][3])), (255, 0, 0), 2)

        cv2.imwrite(output_dir + i.replace('.jpg', '_result.jpg'), img)


infer_ssh()
