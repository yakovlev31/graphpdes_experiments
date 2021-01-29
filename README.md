# Description

Code for reproducing experiments from [Learning continuous-time PDEs from sparse data with graph neural networks](https://openreview.net/forum?id=aUX5Plaq7Oy).

Data can be downloaded [here](https://drive.google.com/file/d/1K5LsXU8JRln5OSJ049jims4v98l2U9cm/view?usp=sharing).

You will need to install [graphpdes](https://github.com/yakovlev31/graphpdes) before running the scripts.

The data comes in the following format:

* t (time grid) ndarray with shape (simulations, timepoints)
* x (node positions) ndarray with shape (simulations, nodes, grid dimension)
* u (field) ndarray with shape (simulations, timepoints, nodes, field dimension)
* bcs_dicts (only for Dirichlet BCs) - dictionary {'bc_name': [[bc_node_indices], [field_indices]], etc}. Used to zero the predicted time derivative of u at nodes in bc_node_indices and field dimensions in field_indices.
