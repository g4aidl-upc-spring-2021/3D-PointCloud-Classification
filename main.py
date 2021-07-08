"""
Usage:
  main.py [--numFeatures=<k>] [--numClasses=<nc>] [--learningRate=<lr>] [--batchSize=<bs>] [--numWorkers=<nw>] [--shuffleTrain=<st>] [--shuffleValid=<sv>] [--epochs=<e>] [--schedule=<sch>] [--debug=<db>] [--accuracyAverage=<aavg>] [--patience=<p>] [--absentScore=<as>] [--tbLogs=<tbl>]
  main.py -h | --help
Options:
  --numFeatures=<k>       Number of features [default: 3]
  --numClasses=<nc>       Number of segmentation classes [default: 50]
  --learningRate=<lr>     Learning Rate [default: 1e-3]
  --batchSize=<bs>        Batch Size[default: 32]
  --numWorkers=<nw>       Number of workers [default: 2]
  --shuffleTrain=<st>      Shuffle Train [default: False]
  --shuffleValid=<sv>      Shuffle Validation [default: False]
  --epochs=<e>            Number of epochs [default: 10]
  --schedule=<sch>        Schedule [default: False]
  --debug=<db>            Debug [default: True]
  --accuracyAverage=<aavg> Accuracy Average [default: 'micro']
  --patience=<p>          Patience [default: 5]
  --absentScore=<as>      Absent Score [default: 1]
  --tbLogs=<tbl>          Dir to save tensorboard logs [default: './tensorboard/']

"""
import os
import datetime
import numpy as np

import tensorflow
import tensorboard

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import DataLoader

from torchmetrics import Accuracy



from docopt import docopt

from dataset import get_train_valid_test_ModelNet, get_data_augmentation
from Models.PointNet import PointNetModel
from Models.GCN import GCN
from utils import my_print


hparams = {
    'k': 3,
    'num_classes': 10,
    'lr': 1e-3,
    'bs': 32,
    'num_workers': 2,
    'epochs': 100,
    'schedule': False,
    'debug': True,
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'tb_logs': './tensorboard/',
    'tb_name': 'tb_' + str(datetime.datetime.utcnow()),
    'model': 'pointNet',
    'level': 3,
    'dropout': True,
    'data_augmentation': 'flip_rotate',
    'wd': 1e-3,
    'max_lr': 100,
    'gamma': 0.5,
    'patience': 10,
    'scheduler': 'CyclicLR',
    'step_size': 20
}


def get_model(model_name, k=3, num_classes=10, dropout=True, level=3):
    if model_name.lower() == 'pointNet'.lower():
        return PointNetModel(k, num_classes, dropout)
    elif model_name.upper() == 'GCN':
        return GCN(k, num_classes, level, dropout)
    else:
        raise ValueError('Model is not correctly introduced')


def get_scheduler(scheduler_name, optimizer, lr=1e-3, max_lr=1e2, gamma=0.5, patience=10, step_size=20):
    if scheduler_name.lower() == 'StepLR'.lower():
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma), None
    elif scheduler_name.lower() == 'ReduceLROnPlateau'.lower():
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, patience=patience), None
    elif scheduler_name.lower() == 'CyclicLR'.lower():
        return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=max_lr, cycle_momentum=False), \
               torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=max_lr, cycle_momentum=False)
    else:
        my_print('No scheduler', hparams['debug'])
        return None, None


def get_tensorflow_writer(root):
    tensorflow.io.gfile = tensorboard.compat.tensorflow_stub.io.gfile  # avoid tensorboard crash when adding embeddings
    train_log_dir = os.path.join(root, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 'train')
    valid_log_dir = os.path.join(root, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 'valid')
    train_writer = SummaryWriter(log_dir=train_log_dir)
    valid_writer = SummaryWriter(log_dir=valid_log_dir)
    return train_writer, valid_writer


def fit(train_data, valid_data, num_classes, k=3, bs=32, num_epochs=100, lr=1e-3, wd=1e-3, num_workers=2):
    # Data Loaders for train and validation:
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=bs, shuffle=False)

    # Obtain correct model
    model = get_model(hparams['model'], k, num_classes, hparams['dropout'], hparams['level']).to(hparams['device'])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Obtain correct scheduler
    scheduler, train_scheduler, fit_scheduler = get_scheduler(hparams['scheduler'], optimizer, lr, hparams['max_lr'], hparams['gamma'],
                              hparams['patience'], hparams['step_size'])
    criterion = nn.CrossEntropyLoss().to(hparams['device'])

    # Metric
    accuracy = Accuracy(average='micro', compute_on_step=False).to(hparams['device'])

    # Tensorboard set up
    train_writer, valid_writer = get_tensorflow_writer(hparams['tb_logs'])

    # Minimum accuracy to save the model
    best_accuracy = 0.0
    my_print('Start training...', hparams['debug'])
    for epoch in range(1, num_epochs + 1):
        my_print('Epoch: ' + str(epoch), hparams['debug'])
        # Train and validation
        train_loss, train_accu = train_epoch(model, train_loader, optimizer, criterion, accuracy, hparams['device'],
                                             train_scheduler)
        valid_loss, valid_accu = valid_epoch(model, valid_loader, criterion, accuracy, hparams['device'])



if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)
    """
    hparams = {
    'k': int(args['--numFeatures']),
    'num_classes': int(args['--numClasses']),
    'lr': float(args['--learningRate']),
    'bs': int(args['--batchSize']),
    'num_workers': int(args['--numWorkers']),
    'shuffle_train': strtobool(args['--shuffleTrain']),
    'shuffle_valid': strtobool(args['--shuffleValid']),
    'epochs': int(args['--epochs']),
    'schedule': strtobool(args['--schedule']),
    'debug': strtobool(args['--debug']),
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'acc_avg': args['--accuracyAverage'],
    'patience': int(args['--patience']),
    'absent_score': int(args['--absentScore']),
    'tb_logs': args['--tbLogs'],
    'tb_name': 'tb_' + str(datetime.datetime.utcnow())
    }
    """

    train_dataset, valid_dataset, test_dataset = get_train_valid_test_ModelNet('/data')
    get_data_augmentation(train_dataset, hparams['data_augmentation'])

    seed = 42
    # Controlling sources of randomness
    torch.manual_seed(seed)  # generate random numbers for all devices (both CPU and CUDA)
    # Random number generators in other libraries:
    np.random.seed(seed)
    # CUDA convolution benchmarking: ensures that CUDA selects the same algorithm each time an application is run
    torch.backends.cudnn.benchmark = False



    best_acc, state_dict_root = fit(train_dataset, valid_dataset, hparams['num_classes'], hparams['k'], hparams['bs'],
                                    hparams['epochs'], hparams['lr'], hparams['wd'], hparams['num_workers'])

