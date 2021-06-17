import torch
import numpy as np
import datetime
import os
import tensorflow
import tensorboard
import dataset

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, IoU
from utils import my_print, write_metric_tb
from model import PointNetModel

hparams = {
    'k': 3,
    'num_classes': 50,
    'lr': 1e-3,
    'bs': 32,
    'num_workers': 2,
    'shuffle_train': False,
    'shuffle_valid': False,
    'epochs': 10,
    'schedule': False,
    'debug': True,
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'acc_avg': 'micro',
    'patience': 5,
    'absent_score': 1,
    'tb_logs': './tensorboard/',
    'tb_name': 'tb_' + str(datetime.datetime.utcnow())
}


def train_epoch(our_model, train_loader, optimizer, criterion, accuracy, iou):
    # List for epoch loss:
    epoch_train_loss = []
    # Model in train mode:
    our_model.train()
    # Metrics stored information reset:
    accuracy.reset()
    iou.reset()
    # Batch loop for training:
    for i, data in enumerate(train_loader, 0):
        # Data retrieval from each bath:
        points = data.pos.unsqueeze(2).to(hparams['device'])
        targets = data.y.to(hparams['device'])
        # Forward pass:
        preds, probs = our_model(points)
        # Loss calculation + Backpropagation pass
        optimizer.zero_grad()
        loss = criterion(preds.squeeze(-1), targets)
        epoch_train_loss.append(loss.item())  # save loss to later represent
        loss.backward()
        optimizer.step()
        # Batch metrics calculation:
        accuracy.update(probs.squeeze(), targets)
        iou.update(probs.squeeze(), targets)
    # Mean epoch metrics calculation:
    mean_loss = np.mean(epoch_train_loss)
    mean_acc = accuracy.compute().item()
    mean_iou = iou.compute()
    mean_iou = iou.compute().item()
    # Print of all metrics:
    my_print('Train loss: ' + str(mean_loss) + "| Acc.: " + str(mean_acc) + "| IoU: " + str(mean_iou), hparams['debug'])
    return mean_loss, mean_acc, mean_iou


def valid_epoch(our_model, valid_loader, scheduler, criterion, accuracy, iou):
    # List for epoch loss:
    epoch_valid_loss = []
    # Model in validation (evaluation) mode:
    our_model.eval()
    # Metrics stored information reset:
    accuracy.reset()
    iou.reset()
    # Batch loop for validation:
    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            # Data retrieval from each bath:
            points = data.pos.unsqueeze(2).to(hparams['device'])
            targets = data.y.to(hparams['device'])
            # Forward pass:
            preds, probs = our_model(points)
            # Loss calculation + Backpropagation pass
            loss = criterion(preds.squeeze(-1), targets)
            epoch_valid_loss.append(loss.item())
            # Batch metrics calculation:
            accuracy.update(probs.squeeze(), targets)
            iou.update(probs.squeeze(), targets)
    # Mean epoch metrics calculation:
    mean_loss = np.mean(epoch_valid_loss)
    mean_acc = accuracy.compute().item()
    mean_iou = iou.compute().item()
    # Learning rate adjustment with scheduler:
    if hparams['schedule']:
        scheduler.step(mean_loss)
    # Print of all metrics:
    my_print('Valid loss: ' + str(mean_loss) + "| Acc.: " + str(mean_acc) + "| IoU: " + str(mean_iou), hparams['debug'])
    return mean_loss, mean_acc, mean_iou


def train_all_epochs(train_loader, valid_loader, our_model, optimizer, scheduler, criterion, accuracy, iou,
                     writer_train, writer_val):
    for epoch in range(hparams['epochs']):
        my_print("Epoch: " + str(epoch), hparams['debug'])
        # Training epoch:
        epoch_train_loss, epoch_train_macc, epoch_train_miou = train_epoch(our_model, train_loader, optimizer,
                                                                           criterion, accuracy, iou)
        # Validation epoch:
        epoch_valid_loss, epoch_valid_macc, epoch_valid_miou = valid_epoch(our_model, valid_loader, scheduler,
                                                                           criterion, accuracy, iou)

        metrics_train = {
            'epoch_train_loss': epoch_train_loss,
            'epoch_train_macc': epoch_train_macc,
            'epoch_train_miou': epoch_train_miou
        }
        write_metric_tb(writer_train, metrics_train, epoch)

        metrics_val = {
            'epoch_valid_loss': epoch_valid_loss,
            'epoch_valid_macc': epoch_valid_macc,
            'epoch_valid_miou': epoch_valid_miou
        }
        write_metric_tb(writer_val, metrics_val, epoch)


if __name__ == '__main__':
    train_dataloader = dataset.get_dataloader(path='data/shapenet', split="train", bs=hparams['bs'],
                                              shuffle=hparams['shuffle_train'], num_workers=hparams['num_workers'])
    valid_dataloader = dataset.get_dataloader(path='data/shapenet', split="val", bs=hparams['bs'],
                                              shuffle=hparams['shuffle_valid'], num_workers=hparams['num_workers'])
    # Features
    our_model = PointNetModel(k=hparams['k'], num_classes=hparams['num_classes'])
    optimizer = torch.optim.Adam(our_model.parameters(), lr=hparams['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=hparams['patience'])
    criterion = nn.CrossEntropyLoss().to(hparams['device'])

    # Metrics
    accuracy = Accuracy(num_classes=hparams['num_classes'], average=hparams['acc_avg'], compute_on_step=False). \
        to(hparams['device'])
    iou = IoU(num_classes=hparams['num_classes'], absent_score=hparams['absent_score'], compute_on_step=False). \
        to(hparams['device'])

    # TensorBoard
    tensorflow.io.gfile = tensorboard.compat.tensorflow_stub.io.gfile
    writer_train = SummaryWriter(log_dir=os.path.join(hparams['tb_logs']))
    writer_val = SummaryWriter(log_dir=os.path.join(hparams['tb_logs']))

    # Begin Train
    train_all_epochs(train_dataloader, valid_dataloader, our_model, optimizer, scheduler, criterion, accuracy, iou,
                     writer_train, writer_val)


