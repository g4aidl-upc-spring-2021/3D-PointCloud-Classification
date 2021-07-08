import torch
from torch import nn
from torch.nn import funtional as F


class TNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k

        self.Conv1 = nn.Conv1d(in_channels=k, out_channels=64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.Conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.Conv3 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.FC1 = nn.Linear(in_features=1024, out_features=512)
        self.bn4 = nn.BatchNorm1d(512)
        self.FC2 = nn.Linear(in_features=512, out_features=256)
        self.bn5 = nn.BatchNorm1d(256)

        self.FC3 = nn.Linear(in_features=256, out_features=k * k)

    def forward(self, cloud_points):
        bs = cloud_points.size(0)

        x = F.relu(self.bn1(self.Conv1(cloud_points)))
        x = F.relu(self.bn2(self.Conv2(x)))
        x = F.relu(self.bn3(self.Conv3(x)))

        # size: [batch size, 1024, # of points]
        x = nn.MaxPool1d(x.size(-1))(x)  # pool with kernel = # of points/batch
        # size: [batch size, 1024, 1]
        x = x.view(bs, -1)  # flatten to get horizontal vector

        # size: [batch size, 1024]
        x = F.relu(self.bn4(self.FC1(x)))
        x = F.relu(self.bn5(self.FC2(x)))

        # diagonal matrices initialized, as many as batch size
        init_matrix = torch.eye(self.k, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            init_matrix = init_matrix.cuda()  # gets updated according to f.c. output
        matrix = self.FC3(x).view(-1, self.k, self.k) + init_matrix
        return matrix


class Transform(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.input_transform = TNet(k)
        self.feature_transform = TNet(k=64)

        self.Conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1)
        self.Conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.Conv3 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        bs = x.size(0)

        matrix3x3 = self.input_transform(x)
        x = torch.bmm(torch.transpose(x, 1, 2), matrix3x3).transpose(1, 2)
        x = F.relu(self.bn1(self.Conv1(x)))

        matrix64x64 = self.feature_transform(x)
        x = torch.bmm(torch.transpose(x, 1, 2), matrix64x64).transpose(1, 2)
        x = F.relu(self.bn2(self.Conv2(x)))
        x = self.bn3(self.Conv3(x))

        x = nn.MaxPool1d(x.size(-1))(x)
        global_features = x.view(bs, -1)
        return global_features


class PointNetModel(nn.Module):
    def __init__(self, k=3, num_classes=16, dropout=True):
        super().__init__()
        self.transform = Transform(k)

        self.FC1 = nn.Linear(in_features=1024, out_features=512)
        self.bn1 = nn.BatchNorm1d(512)
        self.FC2 = nn.Linear(in_features=512, out_features=256)
        self.bn2 = nn.BatchNorm1d(256)
        self.FC3 = nn.Linear(in_features=256, out_features=num_classes)
        self.dropout = nn.Dropout(p=0.3) if dropout else None

    def forward(self, x):
        global_features = self.transform(x)
        x = F.relu(self.bn1(self.FC1(global_features)))
        x = self.FC2(x)
        # apply dropout if exists
        x = self.dropout(x) if self.dropout is not None else x
        x = F.relu(self.bn2(x))
        output = self.FC3(x)
        return output, F.softmax(output, dim=1)
