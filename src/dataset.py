import os
import torch

from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints, NormalizeScale, RandomFlip, RandomRotate, Compose, KNNGraph


def get_dataset(root, transform, pre_transform):
    train_valid_dataset = ModelNet(root=root, name="10", train=True, pre_transform=pre_transform, transform=transform)
    test_dataset = ModelNet(root=root, name="10", train=False, pre_transform=pre_transform, transform=transform)
    return train_valid_dataset, test_dataset


def get_split(index_file_root, dataset):
    index_file = open(index_file_root, 'r')
    train_index = []
    for idx in index_file:
        train_index.append(int(idx))

    return dataset[train_index]


def create_file_if_necessary(train_file, valid_file, dataset):
    if not os.path.isfile(train_file) and not os.path.isfile(valid_file):
        torch.manual_seed(0)
        # Shuffle before splitting data (random split)
        _, perm = dataset.shuffle(return_perm=True)

        # Create two files with the indices od the training and validation data
        train_idx = open(train_file, 'w+')
        valid_idx = open(valid_file, 'w+')

        # Split the tensor of indices in training and validation
        train_split, val_split = perm.split(round(len(perm) * 0.8))

        for i in range(len(train_split)):
            train_idx.writelines(str(train_split[i].item()) + "\n")
        for i in range(len(val_split)):
            valid_idx.writelines(str(val_split[i].item()) + "\n")

        train_idx.close()
        valid_idx.close()

    elif not os.path.isfile(train_file) or not os.path.isfile(valid_file):
        raise ValueError('One file exists and the other one does not')


def get_train_valid_test_ModelNet(root, num_of_points, isGraph=False, normalize_scale=True):
    dataset_root = os.path.join(root, 'ModelNet')
    train_valid_split, test_split = get_dataset(dataset_root, transform=get_transformation(isGraph, normalize_scale),
                                                pre_transform=get_pre_transformation(num_of_points))

    train_split_root = os.path.join(root, 'train_split.txt')
    valid_split_root = os.path.join(root, 'valid_split.txt')
    create_file_if_necessary(train_split_root, valid_split_root, train_valid_split)

    train_split = get_split(index_file_root=train_split_root, dataset=train_valid_split)
    valid_split = get_split(index_file_root=valid_split_root, dataset=train_valid_split)
    return train_split, valid_split, test_split


def get_transformation(normalize_scale, is_graph=False):
    if normalize_scale:
        if is_graph:
            return Compose([NormalizeScale(), KNNGraph(k=9, loop=True, force_undirected=True)])
        else:
            return NormalizeScale()
    else:
        return None


def get_pre_transformation(number_points=1024):
    return SamplePoints(num=number_points)


def get_random_flip(axis=1, p=0.5):
    return RandomFlip(axis, p)


def get_random_rotation(degrees=45, axis=1):
    return RandomRotate(degrees, axis)


def data_augmentation_flip(normalize_scale, axis=1, p=0.5):
    return Compose([get_transformation(normalize_scale), get_random_flip(axis, p)])


def data_augmentation_rotation(normalize_scale, axis=1, degrees=45):
    return Compose([get_transformation(normalize_scale), get_random_rotation(axis=axis, degrees=degrees)])


def data_augmentation_flip_rotation(normalize_scale, axis_flip=1, p=0.5, axis_rotation=1, degrees=45):
    return Compose([get_transformation(normalize_scale), get_random_flip(axis_flip, p),
                    get_random_rotation(axis=axis_rotation, degrees=degrees)])


def get_data_augmentation(dataset, transformation, normalize_scale, axis_flip=1, p=0.5, axis_rotation=1, degrees=45):
    if transformation.lower() == 'flip_rotation':
        dataset.transform = data_augmentation_flip_rotation(normalize_scale, axis_flip, p, axis_rotation, degrees)
    elif transformation.lower() == 'flip':
        dataset.transform = data_augmentation_flip(normalize_scale, axis=axis_flip, p=p)
    elif transformation.lower() == 'rotate':
        dataset.transform = data_augmentation_rotation(normalize_scale, axis=axis_rotation, degrees=degrees)
