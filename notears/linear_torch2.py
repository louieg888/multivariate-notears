import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
import torch

from scipy.linalg import block_diag
from scipy.special import expit as sigmoid
import torch.nn as nn
from torchsummary import summary


class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True),
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128), nn.ReLU(True), nn.Linear(128, encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True),
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


# enc = Encoder(4)
# dec = Decoder(4)
# # summary(enc, input_size=(1, 28, 28))
# # enc(torch.ones(2,1,28,28))


class ConvolutionalAE(nn.Module):
    def __init__(self, graph, schema, embedding_dim=4):
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(embedding_dim)
        self.embedding_dim = embedding_dim

    @staticmethod
    def infer_data_schema(obs_dict):
        # keys in obs_dict are the node_names
        # non batch dimension of the values are the node_dims
        # return an ordereddict node_names, node_values
        return OrderedDict(
            (node_name, obs_value.shape[1]) for node_name, obs_value in obs_dict.items()
        )

    # for an example batch with 3 samples
    # OrderedDict
    # obs_dict = OrderedDict("thickness":torch.zeros((3,4)), "intensity":torch.ones((3,1))*2)

    def forward(self, obs_dict, images):
        im_embs = self.encoder(images)
        pred_ims = self.decoder(im_embs)

        X = im_embs
        for node_name, node_obs in obs_dict.items():
            X = torch.cat([X, torch.Tensor(node_obs)], axis=1)
        return X


class CausalConvolutionalAE(nn.Module):
    def __init__(self, graph, schema, embedding_dim=4):
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(embedding_dim)
        self.embedding_dim = embedding_dim

    @staticmethod
    def infer_data_schema(obs_dict):
        return OrderedDict(
            (node_name, obs_value.shape[1]) for node_name, obs_value in obs_dict.items()
        )

    def forward(self, obs_dict, images):
        im_embs = self.encoder(images)
        pred_ims = self.decoder(im_embs)

        X = im_embs
        for node_name, node_obs in obs_dict.items():
            X = torch.cat([X, torch.Tensor(node_obs)], axis=1)

        return X


# batch wise no tears
# no tears helper
# Loss should only return a number

# class AELoss:
# DAGsimilaryLoss


class NoTearsLoss(nn.Module):
    def __init__(self, schema, max_iter=100, h_tol=1e-8, rho_max=1e16, w_threshold=0.3):
        self.schema = schema
        self.mask = self._mask()
        self.w_est_len = self._get_w_est_len()
        self.d = self.mask.shape[0]

    def _mask(self):
        """
        Mask the block diagonal for an expanded W matrix with zeros, removes the possibility for self loops
        """
        block_matrices = [np.ones((dim, dim)) for dim in list(self.schema.values())]
        f_mask = block_diag(*block_matrices)
        return torch.tensor(1 - f_mask)

    def _get_w_est_len(self):
        """
        Number of non-zero variable entries in W after masking the block diagonal
        """
        return sum(self.mask)

    @functools.lru_cache(maxsize=100, typed=False)
    def reconstruct_W(w):
        """
        Expand w to block matrix W
        """
        W = torch.zeros(self.d, self.d, dtype=torch.float64)
        nonzero_locations = self.w_est_len
        for ind, tup in enumerate([tuple(val) for val in nonzero_locations]):
            W[tup] = w[ind]
        return W

    @functools.lru_cache(maxsize=100, typed=False)
    def f(W):
        """
        In every block of the expanded W matrix, sum the entries and returns a
        new W with the aggregated flows
        """
        dims = torch.tensor(list(self.schema.values()))  # list of node dimensions
        dims_cumsum = torch.cumsum(dims, dim=0)

        def get_grid(lst):
            grid_xs, grid_ys = torch.meshgrid(lst, lst)
            original_shape = grid_ys.size()
            locations = torch.tensor(
                [(x, y) for x, y in zip((grid_xs.flatten()), grid_ys.flatten())]
            )
            locations = locations.reshape((*original_shape, 2))
            return locations

        locations_grid = get_grid(dims_cumsum)
        dimensions_grid = get_grid(dims)
        index_grid = get_grid(torch.tensor(range(len(dims))))

        def get_sum(index):
            location = locations_grid[tuple(index)]
            dimension = dimensions_grid[tuple(index)]

            loc_x, loc_y = location
            dim_x, dim_y = dimension

            s = torch.sum(W[loc_x - dim_x : loc_x, loc_y - dim_y : loc_y])
            return s

        flattened_index_grid = index_grid.reshape((np.prod(index_grid.size()) // 2, 2))
        f_W_entries = torch.stack([get_sum(entry) for entry in flattened_index_grid])
        f_W = f_W_entries.reshape(index_grid.size()[:2])

        f_W = (1 - torch.eye(f_W.shape[0])) * f_W

        return f_W

    def forward(self, X, W_true):
        # X ???> w_est, W_true
        return X


def non_shitty_contains(i, j):
    for tup in torch.nonzero(1 - mask(data_schema())):
        if torch.all(torch.eq(torch.tensor([i, j]), tup)):
            return True

    return False


# def data_schema():
#     return {
#         "embedding": 4,
#         "intensity": 4,
#     }


def mask(schema):
    block_matrices = [np.ones((dim, dim)) for dim in list(schema.values())]
    f_mask = block_diag(*block_matrices)
    return torch.tensor(1 - f_mask)


def get_w_est_len(schema):
    total_size = sum(list(schema.values())) ** 2
    block_diag_size = sum([dim**2 for dim in list(schema.values())])
    return total_size - block_diag_size


def reconstruct_W(w, schema):
    d = sum(list(schema.values()))
    W = torch.zeros(d, d, dtype=torch.float64)
    nonzero_locations = torch.nonzero(mask(schema))
    for ind, tup in enumerate([tuple(val) for val in nonzero_locations]):
        W[tup] = w[ind]

    return W


def reconstruct_w(W, schema):
    d = sum(list(schema.values())) ** 2
    w = torch.zeros(get_w_est_len(schema))
    nonzero_locations = torch.nonzero(mask(schema))
    for ind, tup in enumerate([tuple(val) for val in nonzero_locations]):
        w[ind] = W[tup]

    return w


def f(W, schema):
    dims = torch.tensor(list(schema.values()))
    dims_cumsum = torch.cumsum(dims, dim=0)

    def get_grid(lst):
        grid_xs, grid_ys = torch.meshgrid(lst, lst)
        original_shape = grid_ys.size()
        locations = torch.tensor(
            [(x, y) for x, y in zip((grid_xs.flatten()), grid_ys.flatten())]
        )
        locations = locations.reshape((*original_shape, 2))
        return locations

    locations_grid = get_grid(dims_cumsum)
    dimensions_grid = get_grid(dims)
    index_grid = get_grid(torch.tensor(range(len(dims))))

    def get_sum(index):
        location = locations_grid[tuple(index)]
        dimension = dimensions_grid[tuple(index)]

        loc_x, loc_y = location
        dim_x, dim_y = dimension

        s = torch.sum(W[loc_x - dim_x : loc_x, loc_y - dim_y : loc_y])
        return s

    flattened_index_grid = index_grid.reshape((np.prod(index_grid.size()) // 2, 2))
    f_W_entries = torch.stack([get_sum(entry) for entry in flattened_index_grid])
    f_W = f_W_entries.reshape(index_grid.size()[:2])

    f_W = (1 - torch.eye(f_W.shape[0])) * f_W

    return f_W


def notears_linear(
    X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e16, w_threshold=0.3
):
    """Solve min_W L(W; X) + lambda1 ???W???_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """

    def _loss(W):
        """Evaluate value and gradient of loss."""
        # M = X @ W
        M = torch.matmul(X, W)
        # M = torch.matmul(X, W * mask(data_schema()))
        if loss_type == "l2":
            R = X - M
            loss = 0.5 / X.shape[0] * (R**2).sum()
            G_loss = -1.0 / X.shape[0] * torch.matmul(X.t(), R)
        elif loss_type == "logistic":
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * torch.matmul(X.t(), (torch.sigmoid(M) - X))
        elif loss_type == "poisson":
            S = torch.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * torch.matmul(X.t(), (S - X))
        else:
            raise ValueError("unknown loss type")
        # return loss, G_loss
        return loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        # fW = f(torch.abs(W))
        fW = f(W * W)
        fd = fW.shape[0]
        # E = slin.expm(W * W)  # (Zheng et al. 2018)
        fE = torch.matrix_exp(fW * fW)
        E = torch.matrix_exp(W * W)
        h = torch.trace(fE) - fd
        # h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2 * mask(data_schema())
        # return h, G_h
        return h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return reconstruct_W(w, schema=data_schema())
        # return torch.tensor((w[:d * d] - w[d * d:]).reshape([d, d]))

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        # loss, G_loss = _loss(W)
        loss = _loss(W)
        # h, G_h = _h(W)
        h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * torch.abs(w).sum()
        # G_smooth = G_loss + (rho * h + alpha) * G_h
        # g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        # return obj, g_obj
        return obj

    n, d = X.shape
    # w_est, rho, alpha, h = torch.zeros(2 * d * d), 1.0, 0.0, torch.inf  # double w_est into (w_pos, w_neg)
    w_est, rho, alpha, h = (
        torch.zeros(get_w_est_len(schema=data_schema())),
        1.0,
        0.0,
        torch.inf,
    )  # double w_est into (w_pos, w_neg)
    w_est.requires_grad = True
    optimizer = torch.optim.LBFGS(
        [w_est], history_size=100, max_iter=4, line_search_fn="strong_wolfe"
    )
    h_lbfgs = []

    # bnds = [(0, 0) if non_shitty_contains(i,j) else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == "l2":
        X = X - torch.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            optimizer.zero_grad()
            objective = _func(w_est)
            objective.backward(retain_graph=True)
            optimizer.step(lambda: _func(w_est))
            h_lbfgs.append(objective.item())

            # sol = sopt.minimize(_func, w_est.detach(), method='L-BFGS-B', jac=True)
            # w_new = sol.x
            h_new = _h(_adj(w_est))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        h = h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break

    W_est = _adj(w_est)
    W_est[torch.abs(W_est) < w_threshold] = 0
    return W_est


if __name__ == "__main__":

    # cd /Users/iriondoc/Documents/MIDL/multivariate-notears
    from notears import utils
    from utils import set_random_seed

    set_random_seed(1)

    n, d, s0, graph_type, sem_type = 100, 8, 20, "ER", "gauss"
    B_true = utils.simulate_dag(d, s0, graph_type)
    B_true = B_true * mask(data_schema()).numpy()
    B_true[:4, 4:8] = 0

    W_true = utils.simulate_parameter(B_true)
    np.savetxt("W_true.csv", W_true, delimiter=",")

    X = torch.tensor(utils.simulate_linear_sem(W_true, n, sem_type))
    np.savetxt("X.csv", X, delimiter=",")

    W_est = notears_linear(X, lambda1=0.1, loss_type="l2")
    # assert utils.is_dag(W_est)
    np.savetxt("W_est.csv", W_est, delimiter=",")
    acc = utils.count_accuracy(B_true, W_est != 0)
    print(acc)
