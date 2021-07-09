import tensorflow
import tensorboard
import datetime
import os

import torch
from torch.utils.tensorboard import SummaryWriter


def my_print(text, debug):
    if debug:
        print(text)


def get_tensorflow_writer(root):
    tensorflow.io.gfile = tensorboard.compat.tensorflow_stub.io.gfile  # avoid tensorboard crash when adding embeddings
    train_log_dir = os.path.join(root, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 'train')
    valid_log_dir = os.path.join(root, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 'valid')
    train_writer = SummaryWriter(log_dir=train_log_dir)
    valid_writer = SummaryWriter(log_dir=valid_log_dir)
    return train_writer, valid_writer


def write_epoch_data(train_writer, valid_writer, train_loss, valid_loss, train_accuracy, valid_accuracy, epoch):
    # Write Loss and Accuracy in tensorboard:
    train_writer.add_scalar('Loss', train_loss, epoch)
    train_writer.add_scalar('Accu', train_accuracy, epoch)
    valid_writer.add_scalar('Loss', valid_loss, epoch)
    valid_writer.add_scalar('Accu', valid_accuracy, epoch)


def update_best_model(valid_accuracy, model_state_dict, model_root):
    model_path = os.path.join(model_root, datetime.datetime.now().strftime("%Y%m%d%h"))
    torch.save(model_state_dict, model_path + '.pt')
    return valid_accuracy, model_path
