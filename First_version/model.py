import torch
from torch import nn


class TNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.sharedMLP = nn.Sequential(
            nn.Conv1d(in_channels=k, out_channels=64, kernel_size=1, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.FC = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=k * k),
        )

    def forward(self, cloud_points):
        bs = cloud_points.size(0)
        x = self.sharedMLP(cloud_points)
        # size: [batch size, 1024, # of points]
        x = nn.MaxPool1d(x.size(-1))(x)  # pool with kernel = # of points/batch
        # size: [batch size, 1024, 1]
        x = nn.Flatten(start_dim=1)(x)  # flatten to get horizontal vector
        # size: [batch size, 1024]
        x = self.FC(x)
        # diagonal matrices initialized, as many as batch size
        init_matrix = torch.eye(self.k, requires_grad=True).repeat(bs, 1, 1)
        if torch.cuda.is_available():
            init_matrix = init_matrix.cuda()  # gets updated according to f.c. output
        matrix = x.view(-1, self.k, self.k) + init_matrix
        return matrix


class Transform(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        # Input transform + First Shared MLP:
        self.input_transform = TNet(k)
        self.fc1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        # Feature transform + Second Shared MLP:
        self.feature_transform = TNet(k=64)
        self.fc2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

    def forward(self, x):
        n_points = x.size()[2]
        # ------------------------------------------------------------
        matrix3x3 = self.input_transform(x)
        x = torch.bmm(torch.transpose(x, 1, 2), matrix3x3).transpose(1, 2)
        x = self.fc1(x)
        # ------------------------------------------------------------
        matrix64x64 = self.feature_transform(x)
        y = torch.bmm(torch.transpose(x, 1, 2), matrix64x64).transpose(1, 2)
        x = self.fc2(y)
        # Maxpool + Concat + Output:
        x = nn.MaxPool1d(x.size(-1))(x)
        x = nn.Flatten(1)(x).repeat(n_points, 1, 1).transpose(0, 2).transpose(0, 1)
        y = torch.cat((x, y.repeat(1, 1, n_points)), 1)
        return y, matrix3x3, matrix64x64


class PointNetModel(nn.Module):
    def __init__(self, k=3, num_classes=50):
        super().__init__()
        self.k = k
        # Transform + Two Last MLP
        self.transform = Transform()
        self.fc1 = nn.Sequential(
            nn.Conv1d(in_channels=1088, out_channels=512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=num_classes, kernel_size=1)
        )

    def forward(self, x):
        x, matrix3x3, matrix128x128 = self.transform(x)
        x = self.fc1(x)
        output = self.fc2(x)
        return output, nn.functional.softmax(output, dim=1)
