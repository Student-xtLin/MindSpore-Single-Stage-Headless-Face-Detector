import os
import cv2
import time
import numpy as np

from mindspore.nn import SGD, Adam
from mindspore import context, TimeMonitor, Model, load_checkpoint, load_param_into_net
import mindspore.common.dtype as mstype
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint

from src.config import config
from src.ssh_model import SSHModel
from src.network_define import LossNet, WithLossCell, TrainOneStepCell, LossCallBack
from src.dataset import data_to_mindrecord_byte_image, create_wider_dataset


rank = 0
device_num = 1
context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target, device_id=0)

def prepare_wider_dataset():
    """ prepare wider dataset """
    print("Start create dataset!")

    prefix = "wider.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    print("CHECKING MINDRECORD FILES ...")

    if rank == 0 and not os.path.exists(mindrecord_file + ".db"):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)

        if os.path.isdir(config.image_dir) and os.path.exists(config.anno_path):
            if not os.path.exists(config.image_dir):
                print("Please make sure config:image_dir is valid.")
                raise ValueError(config.image_dir)
            print("Create Mindrecord. It may take some time.")
            data_to_mindrecord_byte_image(config, prefix)
            print("Create Mindrecord Done, at {}".format(mindrecord_dir))
        else:
            print("image_dir or anno_path not exits.")

    while not os.path.exists(mindrecord_file + ".db"):
        time.sleep(5)

    print("CHECKING MINDRECORD FILES DONE!")

    dataset = create_wider_dataset(config, mindrecord_file, batch_size=config.batch_size,
                                        device_num=device_num, rank_id=rank,
                                        num_parallel_workers=config.num_parallel_workers,
                                        python_multiprocessing=config.python_multiprocessing)

    dataset_size = dataset.get_dataset_size()
    print("Create dataset done!")

    return dataset_size, dataset


def train_ssh():
    dataset_size, dataset = prepare_wider_dataset()

    net = SSHModel(config)
    net = net.set_train()

    param_dict = load_checkpoint('./vgg.ckpt')                          # 利用预训练的vgg16进行训练。
    # param_dict = load_checkpoint('./ckpts/ckpt_0/ssh_5-10_100.ckpt')  # 在预训练的ssh基础上继续训练
    keys = [key for key in param_dict]
    for k in keys:
        if k.find('reg') != -1 or k.find('cls') != -1:
            param_dict.pop(k)
    load_param_into_net(net, param_dict)

    device_type = "Ascend" if context.get_context("device_target") == "Ascend" else "Others"
    if device_type == "Ascend":
        net.to_float(mstype.float16)

    loss = LossNet()
    # opt = SGD(params=net.trainable_params(), learning_rate=config.lr, momentum=config.momentum,
    #           weight_decay=config.weight_decay, loss_scale=config.loss_scale)
    opt = Adam(params=net.trainable_params(), learning_rate=(config.lr/10), weight_decay=config.weight_decay)
    net_with_loss = WithLossCell(net, loss)
    net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale)

    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossCallBack(per_print_times=dataset_size, rank_id=rank)
    ckptconfig = CheckpointConfig(save_checkpoint_steps=dataset_size,
                                  keep_checkpoint_max=4)
    save_checkpoint_path = os.path.join(config.save_checkpoint_path, "ckpt_" + str(rank) + "/")
    ckpoint_cb = ModelCheckpoint(prefix='ssh', directory=save_checkpoint_path, config=ckptconfig)
    cb = [time_cb, loss_cb, ckpoint_cb]

    model = Model(net)
    model.train(config.epoch_size, dataset, callbacks=cb)


train_ssh()
