import numpy as np
from sklearn.datasets import make_circles, make_moons, make_blobs, make_s_curve, make_swiss_roll
import os


root = "in/toy"
n0, n1 = 10000, 1000
n = n0 + n1
f = 1.5


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
    os.makedirs(root, exist_ok=True)
    XX = np.array(X, dtype=np.float32)
    yy = np.array(y, dtype=np.bool8)
    p = np.random.permutation(len(y))
    np.savez(f"{root}/{name}", x=XX[p], y=yy[p])


# blobs
X0, _ = make_blobs(
    n0, centers=[[-3.0, -13.0], [3.0, 3.0]], cluster_std=[0.5, 1.5])
X1 = make_noise(X0, n1, f=f)
X, y = combine_data(X0, X1)
savez("blobs", X, y)

# circles
X0, _ = make_circles(n0, noise=0.02, factor=0.5)
X1 = make_noise(X0, n1, f=f)
X, y = combine_data(X0, X1)
savez("circles", X, y)

# moons
X0, _ = make_moons(n0, noise=0.02)
X1 = make_noise(X0, n1, f=f)
X, y = combine_data(X0, X1)
savez("moons", X, y)

# s-curve
X0, _ = make_s_curve(n0, noise=0.05)
X1 = make_noise(X0, n1, f=f)
X, y = combine_data(X0, X1)
savez("s-curve", X, y)

# swiss-roll
X0, _ = make_swiss_roll(n0, noise=0.05)
X1 = make_noise(X0, n1, f=f)
X, y = combine_data(X0, X1)
savez("swiss-roll", X, y)
