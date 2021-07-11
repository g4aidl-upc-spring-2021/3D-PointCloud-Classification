# TODO: Add new usages options
"""
Usage:
  main.py [--numFeatures=<k>] [--numClasses=<nc>] [--learningRate=<lr>] [--batchSize=<bs>] [--numWorkers=<nw>] [--shuffleTrain=<st>] [--shuffleValid=<sv>] [--epochs=<e>] [--schedule=<sch>] [--debug=<db>] [--accuracyAverage=<aavg>] [--patience=<p>] [--absentScore=<as>] [--tbLogs=<tbl>]
  main.py -h | --help
Options:
    --batchSize=<bs>        Batch Size[default: 32]
    --debug=<db>            Debug [default: True]
    --epochs=<e>            Number of epochs [default: 100]
    --dataAugmentation
    --numClasses=<nc>       Number of segmentation classes [default: 50]
  --learningRate=<lr>     Learning Rate [default: 1e-3]
  --numFeatures=<k>       Number of features [default: 3]
  --numWorkers=<nw>       Number of workers [default: 2]
  --shuffleTrain=<st>      Shuffle Train [default: False]
  --shuffleValid=<sv>      Shuffle Validation [default: False]

  --schedule=<sch>        Schedule [default: False]

  --accuracyAverage=<aavg> Accuracy Average [default: 'micro']
  --patience=<p>          Patience [default: 5]
  --absentScore=<as>      Absent Score [default: 1]
  --tbLogs=<tbl>          Dir to save tensorboard logs [default: './tensorboard/']

"""
import datetime
import numpy as np

import torch
from torch import nn

from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch

from torchmetrics import Accuracy

from docopt import docopt

from dataset import get_train_valid_test_ModelNet, get_data_augmentation
from Models.PointNet import PointNetModel
from Models.GCN import GCN
from utils import my_print, get_tensorflow_writer, write_epoch_data, update_best_model

hparams = {
    'bs': 32,
    'epochs': 100,
    'debug': True,
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'tb_logs': './tensorboard/',
    'tb_name': 'tb_' + str(datetime.datetime.utcnow()),
    'data_augmentation': 'flip_rotate',
    'model': 'pointNet',
    'model_log': '/models/',
    'k': 3,
    'num_classes': 10,
    'level': 3,
    'dropout': True,
    'optimizer': 'Adam',
    'lr': 1e-3,
    'wd': 1e-3,
    'momentum': 0.9,
    'scheduler': 'OneCycleLR',
    'gamma': 0.5,
    'patience': 10,
    'step_size': 20
}


def get_model(model_name, k=3, num_classes=10, dropout=True, level=3):
    if model_name.lower() == 'pointNet'.lower():
        return PointNetModel(k, num_classes, dropout)
    elif model_name.upper() == 'GCN':
        return GCN(k, num_classes, level, dropout)
    else:
        raise ValueError('Model is not correctly introduced')


def get_scheduler(scheduler_name, optimizer, lr=1e-3, gamma=0.5, patience=10, step_size=20, train_loader_len=1024,
                  num_epochs=100):
    if scheduler_name.lower() == 'StepLR'.lower():
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma), None, True

    elif scheduler_name.lower() == 'ReduceLROnPlateau'.lower():
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, patience=patience), None, False
    elif scheduler_name.lower() == 'OneCycleLR'.lower():
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, steps_per_epoch=train_loader_len,
                                                        epochs=num_epochs)
        return scheduler, scheduler, None
    else:
        my_print('No scheduler', hparams['debug'])
        return None, None, None


def get_optimizer(optimizer_name, model_parameters, lr, wd, momentum):
    if optimizer_name.lower() == "Adam".lower():
        return torch.optim.Adam(model_parameters, lr=lr, weight_decay=wd)
    elif optimizer_name.upper() == "SGD":
        return torch.optim.SGD(model_parameters, lr=lr, momentum=momentum)
    else:
        raise ValueError('Optimizer is not correctly introduced')


# Train One Epoch:
def train_epoch(model, train_loader, optimizer, criterion, accuracy, device, scheduler):
    # Model in train mode:
    model.train()
    # List for epoch loss:
    epoch_train_loss = []
    # Metric stored information reset:
    accuracy.reset()

    # Train epoch loop:
    for i, data in enumerate(train_loader, 1):
        # Data retrieval from each bath:
        points = to_dense_batch(data.pos, batch=data.batch)[0].to(device).float().transpose(1, 2)
        targets = data.y.to(device)

        # Forward pass:
        preds, probs = model(points)
        # Loss calculation + Backpropagation pass
        optimizer.zero_grad()
        loss = criterion(preds.to(device), targets)
        epoch_train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        # Step for OneCycle scheduler
        if scheduler is not None:
            scheduler.step()

        # Batch metrics calculation:
        accuracy.update(probs, targets)

    # Mean epoch metrics calculation:
    mean_loss = np.mean(epoch_train_loss)
    mean_accu = accuracy.compute().item()
    # Print of all metrics:
    my_print('Train loss: ' + str(mean_loss) + "| Acc.: " + str(mean_accu), hparams['debug'])
    return mean_loss, mean_accu


# Valid One Epoch:
def valid_epoch(model, valid_loader, criterion, accuracy, device):
    # Model in validation (evaluation) mode:
    model.eval()
    # List for epoch loss:
    epoch_valid_loss = []
    # Metric stored information reset:
    accuracy.reset()
    # Batch loop for validation:
    with torch.no_grad():
        for i, data in enumerate(valid_loader, 1):
            # Data retrieval from each bath:
            points = to_dense_batch(data.pos, batch=data.batch)[0].to(device).float().transpose(1, 2)
            targets = data.y.to(device)

            # Forward pass:
            preds, probs = model(points)
            # Loss calculation
            loss = criterion(preds.to(device), targets)
            epoch_valid_loss.append(loss.item())
            # Batch metrics calculation:
            accuracy.update(probs, targets)
    # Mean epoch metrics calculation:
    mean_loss = np.mean(epoch_valid_loss)
    mean_accu = accuracy.compute().item()
    # Print of all metrics:
    print('Valid loss: ', mean_loss, "| Acc.: ", mean_accu)
    return mean_loss, mean_accu


def fit(train_data, valid_data, num_classes, k=3, bs=32, num_epochs=100, lr=1e-3):
    # Data Loaders for train and validation:
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=bs, shuffle=False)

    # Obtain correct model
    model = get_model(hparams['model'], k, num_classes, hparams['dropout'], hparams['level']).to(hparams['device'])

    optimizer = get_optimizer(hparams['optimizer'], model.parameters(), lr, hparams['wd'], hparams['momentum'])

    # Obtain correct scheduler
    scheduler, train_scheduler, fit_scheduler = get_scheduler(hparams['scheduler'], optimizer, lr, hparams['gamma'],
                                                              hparams['patience'], hparams['step_size'],
                                                              len(train_loader), num_epochs)
    criterion = nn.CrossEntropyLoss().to(hparams['device'])

    # Metric
    accuracy = Accuracy(average='micro', compute_on_step=False).to(hparams['device'])

    # Tensorboard set up
    train_writer, valid_writer = get_tensorflow_writer(hparams['tb_logs'])

    # Minimum accuracy to save the model
    best_accuracy = 0.0
    model_root = None
    my_print('Start training...', hparams['debug'])
    for epoch in range(1, num_epochs + 1):
        my_print('Epoch: ' + str(epoch), hparams['debug'])
        # Train and validation
        train_loss, train_accu = train_epoch(model, train_loader, optimizer, criterion, accuracy, hparams['device'],
                                             train_scheduler)
        valid_loss, valid_accu = valid_epoch(model, valid_loader, criterion, accuracy, hparams['device'])

        if fit_scheduler is not None:
            scheduler.step(valid_loss) if fit_scheduler else scheduler.step()

        write_epoch_data(train_writer, valid_writer, train_loss, valid_loss, train_accu, valid_accu, epoch)

        # Save best model:
        if best_accuracy < valid_accu:
            best_accuracy, model_root = update_best_model(valid_accu, model.state_dict())

    final_state_dict_root = model_root + '.pt'
    model.load_state_dict(torch.load(final_state_dict_root))
    my_print("Best accuracy: " + best_accuracy, hparams['debug'])
    return best_accuracy, final_state_dict_root


def test(test_data, model_state_dict_root):
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    model = PointNetModel(k=3, num_classes=10).to(hparams['device'])

    accuracy = Accuracy(average='micro', compute_on_step=False).to(hparams['device'])
    model.load_state_dict(torch.load(model_state_dict_root))
    model.eval()
    # Metric stored information reset:
    accuracy.reset()
    # Batch loop for validation:
    with torch.no_grad():
        for i, data in enumerate(test_loader, 1):
            # Data retrieval from each bath:
            points = to_dense_batch(data.pos, batch=data.batch)[0].to(hparams['device']).float().transpose(1, 2)

            targets = data.y.to(hparams['device'])
            # Forward pass:
            preds, probs = model(points)

            # Batch metrics calculation:
            accuracy.update(probs, targets)

    mean_accu = accuracy.compute().item()
    # Print of all metrics:
    print("Acc.: ", mean_accu)
    return mean_accu


if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)
    """
    hparams = {
    'bs': int(args['--batchSize']),
    'epochs': int(args['--epochs']),
    'debug': strtobool(args['--debug']),
    'tb_name': 'tb_' + str(datetime.datetime.utcnow()),
    'data_augmentation': args['--dataAugmentation'],
    'model': args['--model'],
    'k': int(args['--numFeatures']),
    'num_classes': int(args['--numClasses']),
    'level': int(args['--level']),
    'dropout': strtobool(args['--debug']),
    'optimizer': args['--optimizer'],
    'lr': float(args['--learningRate']),
    'wd': float(args['--weightDecay']),
    'momentum': float(args['--momentum']),
    'scheduler': args['--schedule'],
    'gamma': float(args['gamma']),
    'patience': float(args['--patience']),
    'step_size': int(args['--stepSize'])
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
                                    hparams['epochs'], hparams['lr'])

    test_inference = test(test_dataset, state_dict_root)
