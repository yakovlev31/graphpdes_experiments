import torch
import torch.nn as nn

from torch_geometric.data import DataLoader

import numpy as np

import matplotlib.pyplot as plt

from torchdiffeq import odeint_adjoint as odeint
from graphpdes import Model, DynamicsFunction

from collections import namedtuple

import utils


# can be replaced by argparse
Config = namedtuple(
    "Config", 
    [
        "d", "hs_1", "hs_2", "method", "rtol", 
        "atol", "device", "model_path", "data_path"
    ]
)

args = Config(
    d=40,
    hs_1=60,
    hs_2=0,
    method="adaptive_heun",  # adams
    rtol=0.0,
    atol=1.0e-5,  # 1.0e-7
    device="cuda",
    model_path="./models/model_tmp.pth",
    data_path="./data/convdiff_2pi_n3000_t21_test/",
)

device = torch.device(args.device)

msg_net = nn.Sequential(
    nn.Linear(4, args.hs_1), nn.Tanh(), 
    nn.Linear(args.hs_1, args.hs_1), nn.Tanh(), 
    nn.Linear(args.hs_1, args.hs_1), nn.Tanh(), 
    nn.Linear(args.hs_1, args.d)
)
aggr_net = nn.Sequential(
    nn.Linear(args.d+1, args.hs_1), nn.Tanh(), 
    nn.Linear(args.hs_1, args.hs_1), nn.Tanh(), 
    nn.Linear(args.hs_1, args.hs_1), nn.Tanh(), 
    nn.Linear(args.hs_1, 1)
)

model = Model(aggr_net, msg_net)
F = DynamicsFunction(model).to(device)
F.load_state_dict(torch.load(args.model_path, map_location=device))

data = utils.read_pickle(['t', 'x', 'u'], args.data_path)
dataset = utils.generate_torchgeom_dataset(data)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Loss
loss_fn = nn.MSELoss()

# Testing
diffs_over_time = []
losses = torch.zeros(len(loader))

inds_of_sims_to_show = set([0])

with torch.no_grad():
    for i, dp in enumerate(loader):
        edge_index = dp.edge_index
        pos = dp.pos
        rel_pos = pos[edge_index[1]] - pos[edge_index[0]]
        params_dict = {'edge_index': edge_index.to(device), 'rel_pos': rel_pos.to(device)}
        F.update_params(params_dict)

        y0 = dp.x.to(device)
        t = dp.t.to(device)
        y_pd = odeint(F, y0, t, method=args.method, rtol=args.rtol, atol=args.atol)
        y_gt = dp.y.transpose(0, 1).to(device)
        
        loss = loss_fn(y_pd, y_gt)
        losses[i] = loss.item()

        u = y_gt.cpu().detach().numpy()
        u_pd = y_pd.cpu().detach().numpy()
        u_mean = u.mean(axis=1).reshape(-1)
        
        eps = 1.0e-6
        diffs = [np.linalg.norm(u[i].reshape(-1) - u_pd[i].reshape(-1)) / (np.linalg.norm(u[i].reshape(-1)) + eps) for i in range(len(u))]
        diffs_over_time.append(diffs)

        print("test case {:>5d} | test loss: {:>7.12f}".format(i, losses[i]))

        if i in inds_of_sims_to_show:
            print("Plotting...")
            utils.plot_grid(dataset[i].pos.cpu().detach().numpy())
            plt.figure(0)
            utils.plot_fields(
                t=dataset[i].t,
                coords=dataset[i].pos,
                fields={
                    "y_pd": u_pd,
                    "y_gt": u,
                },
                save_path="./tmp_figs/",
                delay=0.0001,
            )
            plt.show()

        # if i == 2:  # 3 for grids, 2 for time points
        #     break

print("Plotting diffs...")
plt.figure(0)
t = dataset[0].t.numpy()

for diff in diffs_over_time:
    plt.plot(t, diff, alpha=0.5)

plt.plot(t, np.mean(diffs_over_time, axis=0), '--k')

plt.ylabel("Rel. diff.")
plt.xlabel("t (sec)")

# plt.show()
plt.savefig("diffs.png")

diffs_over_time = np.array(diffs_over_time)
print("diffs_over_time.shape", diffs_over_time.shape)
print("diffs_over_time.mean", diffs_over_time.mean())
print("diffs_over_time.mean", diffs_over_time.mean(axis=0))
