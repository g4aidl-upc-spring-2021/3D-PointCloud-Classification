from Models import GCN, PointNet
import numpy as np


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def get_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


point_net_model = PointNet.PointNetModel()
gcn_model = GCN.GCN()

print('PointNet total parameters:', get_n_params(point_net_model))
print('GCN total parameters:', get_n_params(gcn_model))
print('PointNet trainable parameters:', get_trainable_params(point_net_model))
print('GCN trainable parameters:', get_trainable_params(gcn_model))
