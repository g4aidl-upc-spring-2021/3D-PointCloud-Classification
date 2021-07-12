import tensorflow
import tensorboard
import datetime
import os
import plotly.graph_objects as go

import torch
from torch.utils.tensorboard import SummaryWriter


def my_print(text, debug):
    if debug:
        print(text)


def get_tensorboard_writer(root):
    tensorflow.io.gfile = tensorboard.compat.tensorflow_stub.io.gfile  # avoid tensorboard crash when adding embeddings
    train_log_dir = os.path.join(root, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 'train')
    valid_log_dir = os.path.join(root, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 'valid')
    train_writer = SummaryWriter(log_dir=train_log_dir)
    valid_writer = SummaryWriter(log_dir=valid_log_dir)
    return train_writer, valid_writer


def write_epoch_data(train_writer, valid_writer, train_loss, valid_loss, train_accuracy, valid_accuracy, epoch):
    # Write Loss and Accuracy in tensorboard:
    train_writer.add_scalar('Loss', train_loss, epoch)
    train_writer.add_scalar('Accu', train_accuracy, epoch)
    valid_writer.add_scalar('Loss', valid_loss, epoch)
    valid_writer.add_scalar('Accu', valid_accuracy, epoch)


def update_best_model(valid_accuracy, model_state_dict, model_root):
    model_path = os.path.join(model_root, datetime.datetime.now().strftime("%Y%m%d%h"))
    torch.save(model_state_dict, model_path + '.pt')
    return valid_accuracy, model_path


def visualize_point_cloud(point_cloud):
    points, y = point_cloud
    fig = go.Figure(data=[go.Mesh3d(x=points[1][:, 0], y=points[1][:, 1], z=points[1][:, 2],
                                    mode='markers', marker=dict(size=3, opacity=1))])
    fig.show()


def visualize_graph_point_cloud(point_cloud):
    edge_index, points, y = point_cloud
    edge_x = [], edge_y = [], edge_z = []
    edges_index = edge_index[1].numpy().T
    real_points = points[1].numpy()

    # Get coordinates from adjacency matrix
    for i, edge in enumerate(edges_index):
        x0, y0, z0 = real_points[edge[0]]
        x1, y1, z1 = real_points[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = [], node_y = [], node_z = []
    # Get node coordinates
    for node in real_points:
        x, y, z = node
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # color scale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Network graph made with Python',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.show()