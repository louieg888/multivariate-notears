import torch.nn as nn
import torch

"""
X

w_est (learned directly, or assembled from something that's learned)

f(W_true)
node level view
    a   b  c  d
a  [ 0, 0, 1, 0,
b    1, 0, 0, 0,
c    0, 0, 0, 1,
d    0, 0, 0, 0,]

W_true
expanded block level
    a  b0 b1  c  d
a  [ 0, 0, 0, 1, 0,
b0   1, 0, 0, 0, 0,
b1   1, 0, 0, 0, 0,
c    0, 0, 0, 0, 1,
d    0, 0, 0, 0, 0,]

b0 –> a
b1 –> a
a –> c
c –> d

w_true = [0, 0, 1, 0, 1, 0, 0, 0, ...]

w_est = [-0.2, 1.0, 0.34, 0.3, ....] in R

"""
# 4*4-4
# get_w_est_len
w_est = torch.random_normal(4 * 4 - 4, requires_grad=True)  # one dimensional


def loss1(W_true, trainable_w):
    """
    similarity between learned w_est and known binary W_true
    """
    W_est = reconstruct_W(w_est)
    # reconstruct_W
    W_est_abs = torch.abs(W_est)
    W_est_abs_binary = torch.sigmoid(W_est_abs)
    #
    return torch.sum(torch.square(W_true - W_est_abs_binary))


def loss2(W_est):
    """
    DAGness, aka no acyclicity is allowed
    """
    return _h(W_est)


def loss3(X, W_est):
    """
    Reconstructing the data from the learned linear (or not!) relationships
    """
    X_est = torchtorch.matmul(W_est, X)

    return
