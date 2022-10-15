import numpy as np
from sklearn.datasets import make_circles, make_moons, make_blobs, make_s_curve, make_swiss_roll
import os


def make_noise(X0: np.ndarray, n1: int, f=1.0):
    d = X0.shape[1]
    lb = X0.min(axis=0)
    r = X0.ptp(axis=0)
    c0 = lb + 0.5 * r
    c1 = 0.5 * f * r
    n = np.random.rand(n1, d) * f * r
    return n + (c0 - c1)


def combine_data(X0: np.ndarray, X1: np.ndarray):
    X = np.vstack([X0, X1])
    y0 = np.full(len(X0), False)
    y1 = np.full(len(X1), True)
    y = np.hstack([y0, y1])
    return X, y


def savez(name, X, y):
    root = "in/examples"
    os.makedirs(root, exist_ok=True)
    XX = np.array(X, dtype=np.float32)
    yy = np.array(y, dtype=np.bool8)
    p = np.random.permutation(len(y))
    np.savez(f"{root}/{name}", x=XX[p], y=yy[p])


X0, _ = make_blobs(10000, centers=[[0, 10], [10, 0]], cluster_std=[1, 1])
X1, _ = make_blobs(50, centers=[[0, 0], [10, 10]], cluster_std=[1, 1])
X, y = combine_data(X0, X1)
savez("two_blobs", X, y)
