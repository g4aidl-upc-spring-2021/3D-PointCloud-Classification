"""
Usage:
  main.py [--batchSize=<bs>] [--debug=<db>] [--epochs=<e>] [--dataAugmentation=<da>] [--flipProbability=<fp>]
  [--flipAxis=<fa>] [--rotateDegrees=<rd>] [--rotateAxis=<ra>] [--model=<mod>] [--numFeatures=<k>] [--numClasses=<nc>]
  [--level=<lev>]  [--dropout=<do>] [--optimizer=<opt>] [--learningRate=<lr>] [--weightDecay=<wd>] [--momentum=<mom>]
  [--schedule=<sch>] [--gamma=<gam>] [--patience=<p>] [--stepSize=<ss>]

  main.py -h | --help
Options:
    --batchSize=<bs>        Batch Size[default: 32]
    --debug=<db>            Debug [default: True]
    --epochs=<e>            Number of epochs [default: 100]
    --dataAugmentation=<da> Type of data augmentation applied [default: flip_rotation]
    --flipProbability=<fp>  Probability that node positions will be flipped. [default: 0.5]
    --flipAxis=<fa>         The axis along the position of nodes being flipped. [default: 1]
    --rotateDegrees=<rd>    Rotation interval from which the rotation angle is sampled. [default: 45]
    --rotateAxis=<ra>       The rotation axis. [default: 1]
    --model=<mod>           Model that is going to be used [default: PointNet]
    --numFeatures=<k>       Number of features [default: 3]
    --numClasses=<nc>       Number of segmentation classes [default: 50]
    --level=<lev>           Number of layers of GCN [default: 3]
    --dropout=<do>          Dropout probability if used [default: 0.3]
    --optimizer=<opt>       Optimizer that is going to be used [default: Adam]
    --learningRate=<lr>     Learning Rate [default: 1e-3]
    --weightDecay=<wd>      Weight Decay for Adam [default: 1e-3]
    --momentum=<mom>        Momentum for SGD [default: 0.9]
    --schedule=<sch>        Schedule [default: False]
    --gamma=<gam>           Multiplicative factor of learning rate decay. [default:0.5]
    --patience=<p>          Number of epochs with no improvement after which learning rate will be reduced. [default: 5]
    --stepSize=<ss>         Period of learning rate decay. [default: 20]

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
    'flip_probability': 0.5,
    'flip_axis': 1,
    'rotate_degrees': 45,
    'rotate_axis': 1,
    'model': 'pointNet',
    'model_log': './saved_models/',
    'k': 3,
    'num_classes': 10,
    'level': 3,
    'dropout': 0.3,
    'optimizer': 'Adam',
    'lr': 1e-3,
    'wd': 1e-3,
    'momentum': 0.9,
    'scheduler': 'OneCycleLR',
    'gamma': 0.5,
    'patience': 10,
    'step_size': 20
}


def check_if_graph(model_name):
    return model_name.upper() == 'GCN'


def get_model(model_name, k=3, num_classes=10, dropout=0.3, level=3):
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


def get_points(points, dense_batch, device):
    if dense_batch:
        return to_dense_batch(points.pos, batch=points.batch)[0].to(device).float().transpose(1, 2)
    else:
        return points.pos.to(device)


# Train One Epoch:
def train_epoch(model, train_loader, optimizer, criterion, accuracy, device, scheduler, dense_batch):
    # Model in train mode:
    model.train()
    # List for epoch loss:
    epoch_train_loss = []
    # Metric stored information reset:
    accuracy.reset()

    # Train epoch loop:
    for i, data in enumerate(train_loader, 1):
        # Data retrieval from each bath:
        points = get_points(data, dense_batch, device)
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
def valid_epoch(model, valid_loader, criterion, accuracy, device, dense_batch):
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
            points = get_points(data, dense_batch, device)
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
    my_print('Valid loss: ' + str(mean_loss) + "| Acc.: " + str(mean_accu), hparams['debug'])
    return mean_loss, mean_accu


def fit(train_data, valid_data, num_classes, k=3, bs=32, num_epochs=100, lr=1e-3, dense_batch=False):
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
                                             train_scheduler, dense_batch)
        valid_loss, valid_accu = valid_epoch(model, valid_loader, criterion, accuracy, hparams['device'], dense_batch)

        if fit_scheduler is not None:
            scheduler.step(valid_loss) if fit_scheduler else scheduler.step()

        write_epoch_data(train_writer, valid_writer, train_loss, valid_loss, train_accu, valid_accu, epoch)

        # Save best model:
        if best_accuracy < valid_accu:
            best_accuracy, model_root = update_best_model(valid_accu, model.state_dict(), hparams['model_log'])

    final_state_dict_root = model_root + '.pt'
    model.load_state_dict(torch.load(final_state_dict_root))
    my_print("Best accuracy: " + best_accuracy, hparams['debug'])
    return best_accuracy, final_state_dict_root


def test(test_data, model_state_dict_root, dense_batch):
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    model = get_model(hparams['model'], hparams['k'], hparams['num_classes'], hparams['dropout'], hparams['level'])

    accuracy = Accuracy(average='micro', compute_on_step=False).to(hparams['device'])
    model.load_state_dict(torch.load(model_state_dict_root))
    model.eval()
    # Metric stored information reset:
    accuracy.reset()
    # Batch loop for validation:
    with torch.no_grad():
        for i, data in enumerate(test_loader, 1):
            # Data retrieval from each bath:
            points = get_points(data, dense_batch, hparams['device'])
            targets = data.y.to(hparams['device'])

            # Forward pass:
            preds, probs = model(points)

            # Batch metrics calculation:
            accuracy.update(probs, targets)

    mean_accu = accuracy.compute().item()
    # Print of all metrics:
    my_print("Acc.: " + str(mean_accu), hparams['debug'])
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
    'flip_probability': float(args['--flipProbability]),
    'flip_axis': int(args['--flipAxis']),
    'rotate_degrees': float(args['--flipDegrees']),
    'rotate_axis': int(args['--rotateAxis']),
    'model': args['--model'],
    'k': int(args['--numFeatures']),
    'num_classes': int(args['--numClasses']),
    'level': int(args['--level']),
    'dropout': strtobool(args['--dropout']),
    'optimizer': args['--optimizer'],
    'lr': float(args['--learningRate']),
    'wd': float(args['--weightDecay']),
    'momentum': float(args['--momentum']),
    'scheduler': args['--schedule'],
    'gamma': float(args['--gamma']),
    'patience': float(args['--patience']),
    'step_size': int(args['--stepSize'])
    }
    """
    model_is_graph = check_if_graph(hparams['model'])
    train_dataset, valid_dataset, test_dataset = get_train_valid_test_ModelNet('/data', model_is_graph)
    get_data_augmentation(train_dataset, hparams['data_augmentation'], hparams['flip_axis'],
                          hparams['flip_probability'], hparams['rotate_axis'], hparams['flip_axis'])

    seed = 42
    # Controlling sources of randomness
    torch.manual_seed(seed)  # generate random numbers for all devices (both CPU and CUDA)
    # Random number generators in other libraries:
    np.random.seed(seed)
    # CUDA convolution benchmarking: ensures that CUDA selects the same algorithm each time an application is run
    torch.backends.cudnn.benchmark = False

    best_acc, state_dict_root = fit(train_dataset, valid_dataset, hparams['num_classes'], hparams['k'], hparams['bs'],
                                    hparams['epochs'], hparams['lr'], not model_is_graph)

    test_inference = test(test_dataset, state_dict_root, not model_is_graph)
