import pickle
import numpy as np
import torch

from torch_geometric.data import Data

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from scipy.spatial import Delaunay


def get_mask(x, domain='unit_square'):
    mask = np.ones((x.shape[0], 1))
    if domain == 'unit_square':
        for i, xi in enumerate(x):
            if xi[0] == 0.0 or xi[0] == 1.0:
                mask[i] = 0.0
            if xi[1] == 0.0 or xi[1] == 1.0:
                mask[i] = 0.0
    elif domain == 'cutout':
        for i, xi in enumerate(x):
            if xi[0] == 0.0 or xi[0] == 1.0:
                mask[i] = 0.0
            if xi[1] == 0.0 or xi[1] == 1.0:
                mask[i] = 0.0
            if xi[0] >= 0.25 and xi[0] <= 0.75 and xi[1] == 0.5:
                mask[i] = 0.0
            if xi[1] >= 0.0 and xi[1] <= 0.5 and (xi[0] == 0.25 or xi[0] == 0.75):
                mask[i] = 0.0
    else:
        print("Wrong domain name.")
    return mask


def read_pickle(keys, path="./"):
    data_dict = {}
    for key in keys:
        with open(path+key+".pkl", "rb") as f:
            data_dict[key] = pickle.load(f)
    return data_dict


def neighbors_from_delaunay(tri):
    """Returns ndarray of shape (N, *) with indices of neigbors for each node.
    N is the number of nodes.
    """
    neighbors_tri = tri.vertex_neighbor_vertices
    neighbors = []
    for i in range(len(neighbors_tri[0])-1):
        curr_node_neighbors = []
        for j in range(neighbors_tri[0][i], neighbors_tri[0][i+1]):
            curr_node_neighbors.append(neighbors_tri[1][j])
        neighbors.append(curr_node_neighbors)
    return neighbors


def is_near(x, y, eps=1.0e-16):
    x = np.array(x)
    y = np.array(y)
    for yi in y:
        if np.linalg.norm(x - yi) < eps:
            return True
    return False


def get_edge_index(x):
    MAX_DIST = 1.0  # 0.04
    tri = Delaunay(x)
    neighbors = neighbors_from_delaunay(tri)
    edge_index = []
    for i, _ in enumerate(neighbors):
        for _, neighbor in enumerate(neighbors[i]):
            if i == neighbor:
                continue
            if np.linalg.norm(x[i] - x[neighbor]) > MAX_DIST:
                continue
            edge = [i, neighbor]
            edge_index.append(edge)
    edge_index = np.array(edge_index).T
    return edge_index


def generate_torchgeom_dataset(data):
    """Returns dataset that can be used to train our model.
    
    Args:
        data (dict): Data dictionary with keys t, x, u, bcs_dicts.
    Returns:
        dataset (list): Array of torchgeometric Data objects.
    """

    n_sims = data['u'].shape[0]
    dataset = []

    for sim_ind in range(n_sims):
        print("{} / {}".format(sim_ind+1, n_sims))
        
        edge_index = get_edge_index(data['x'][sim_ind])
        
        tg_data = Data(
            x=torch.Tensor(data['u'][sim_ind][0, :, :]),
            edge_index=torch.Tensor(edge_index).long(),
            y=torch.Tensor(data['u'][sim_ind]).transpose(0, 1),
            pos=torch.Tensor(data['x'][sim_ind]),
            t=torch.Tensor(data['t'][sim_ind]),
            sim_ind=torch.tensor(sim_ind, dtype=torch.long)
        )
        
        dataset.append(tg_data)

    return dataset


def get_parameters_count(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_params


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data, gain=1.66666667)
        torch.nn.init.zeros_(m.bias.data)


def get_warmup_scheduler(a):
    def scheduler(epoch):
        if epoch <= a:
            return epoch * 1.0 / a
        else:
            return 1.0
    return scheduler


def plot_graph(coords):
    tri = Delaunay(coords)
    plt.triplot(coords[:, 0], coords[:, 1], tri.simplices.copy())
    plt.plot(coords[:, 0], coords[:, 1], 'o')
    plt.hlines(0, 0, 1)
    plt.hlines(1, 0, 1)
    plt.vlines(0, 0, 1)
    plt.vlines(1, 0, 1)


def get_masked_triang(x, y, max_radius):
    triang = mtri.Triangulation(x, y)
    triangles = triang.triangles
    xtri = x[triangles] - np.roll(x[triangles], 1, axis=1)
    ytri = y[triangles] - np.roll(y[triangles], 1, axis=1)
    maxi = np.sqrt(xtri**2 + ytri**2).max(axis=1)
    triang.set_mask(maxi > max_radius)
    return triang


def plot_triang_grid(ax, coords, values):
    x = coords[:, 0]
    y = coords[:, 1]
    triang = get_masked_triang(x, y, max_radius=1.0)
    levels = np.linspace(0.0, 1.0, 11)
    im = ax.tricontourf(triang, values, levels=levels)  # norm=mpl.colors.Normalize(vmin=-0.5, vmax=1.5)
    # ax.triplot(triang, 'ko-', linewidth=0.1, ms=0.5)
    return im


def plot_grid(coords):
    x = coords[:, 0]
    y = coords[:, 1]
   
    triang = get_masked_triang(x, y, max_radius=1.0)

    plt.triplot(triang, 'ko-', linewidth=0.1, ms=0.5)
    plt.savefig("grid.png")


def plot_fields(t, coords, fields, delay=0.1, save_path=None):
    """
    Args:
        t (ndarray): Time points.
        coords (ndarray): Coordinates of nodes.
        fields (dict): keys - field names.
            values - ndarrays with shape (time, num_nodes, 1).
        save_path (str): Path where plot will be saved as save_path/field_name_time.png
    """
    num_fields = len(fields.keys())

    fig, ax = plt.subplots(1, num_fields, figsize=(6*num_fields, 6))
    if num_fields == 1:
        ax = [ax]
    else:
        ax = ax.reshape(-1)

    mappables = [
        plot_triang_grid(
            ax[i], coords,
            fields[list(fields.keys())[i]][0].squeeze()) for i in range(num_fields)]
    [fig.colorbar(im, ax=ax) for im in mappables]

    plt.show(block=False)

    for j, tj in enumerate(t):
        for i, (key, field) in enumerate(fields.items()):
            ax[i].cla()
            mappables[i] = plot_triang_grid(ax[i], coords, field[j].squeeze())
            ax[i].set_aspect('equal')
            ax[i].set_title("Field {:s} at time {:.6f}".format(key, tj))
            
            if save_path is not None:
                plt.savefig(save_path+"{:d}.png".format(j))

        plt.draw()
        plt.pause(delay)


def concatenate_bcs_dicts(bcs_dicts, dataset, sim_inds):
    concat_bcs_dict = {}
    batch_dataset = [dataset[i] for i in sim_inds]
    shifts = [d['x'].shape[0] for d in batch_dataset]
    batch_bcs_dicts = [bcs_dicts[i] for i in sim_inds]
    for k in bcs_dicts[0].keys():  # all dicts have the same keys
        tmp_bc_inds = []
        tmp_field_inds = batch_bcs_dicts[0][k][1]  # all dicts have the same field ids for the given key
        total_shift = 0
        for i, d in enumerate(batch_bcs_dicts):
            tmp_bc_inds.extend(d[k][0] + total_shift)
            total_shift += shifts[i]
        concat_bcs_dict[k] = [
            np.array(tmp_bc_inds, dtype=np.int64), 
            [[ind] for ind in tmp_field_inds]
        ]
    return concat_bcs_dict
