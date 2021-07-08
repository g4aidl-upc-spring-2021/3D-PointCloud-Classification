from torch import nn
from torch.nn import funtional as F
from torch_geometric.nn import GCNConv, global_max_pool


class GCN(nn.Module):
    def __init__(self, k=3, num_classes=10, level=3, dropout=True):
        super().__init__()
        self.k = k
        self.level = level
        self.classes = num_classes
        self.last_hidden_layer = 8 * 2**self.level

        self.conv1 = GCNConv(self.k, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 64)
        self.fc = nn.Linear(self.last_hidden_layer, self.classes)

        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(p=0.3) if dropout else None

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = self.dropout(x) if self.level == 1 and self.dropout is not None else x
        x = F.relu(self.bn1(x))

        if self.level >= 2:
            x = self.conv2(x, edge_index)
            x = self.dropout(x) if self.level == 2 and self.dropout is not None else x
            x = F.relu(self.bn2(x))

        if self.level >= 3:
            x = self.conv3(x, edge_index)
            x = self.dropout(x) if self.level == 3 and self.dropout is not None else x
            x = F.relu(self.bn3(x))

        # 2. Readout layer: Aggregate node embeddings into a unified graph embedding
        x = global_max_pool(x, batch)  # [batch_size=32, hidden_channels=64]

        # 3. Apply a final classifier
        x = self.fc(x)

        return x, F.softmax(x, dim=1)