from torch_geometric.datasets import ShapeNet
from torch_geometric.data import DataLoader


def get_dataset(path, split):
    return ShapeNet(root=path, split=split, include_normals=False)


def get_dataloader(path, split, bs, shuffle, num_workers):
    return DataLoader(get_dataset(path, split), batch_size=bs, shuffle=shuffle, num_workers=num_workers)
