import gzip
import unlzw
import urllib.request
import os
import io
import pandas as pd
import csv
import numpy as np
from scipy.io.arff import loadarff


root = "in/real"
os.makedirs(root, exist_ok=True)


def savez(name, X, y):
    x = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.bool8)
    np.savez(f"{root}/{name}", x=x, y=y)


def parse_uci(buffer, normal, anomalous, delimiter=' '):
    df = pd.read_csv(buffer, delimiter=delimiter, header=None)
    last = df.shape[1] - 1
    keep = df[last].isin(normal + anomalous)
    X = df[keep].drop(columns=[last]).to_numpy()
    y = df[keep][last].isin(anomalous).to_numpy()
    return X, y


# KDDCUP
with urllib.request.urlopen("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz") as resp:
    with gzip.GzipFile(fileobj=resp) as buffer:
        df = pd.read_csv(buffer, header=None)
        last = df.shape[1] - 1
        # - http
        is_http = df[2] == 'http'
        X_http = df[is_http].drop(columns=[1, 2, 3, last]).to_numpy()
        y_http = (df[is_http][last] != 'normal.').to_numpy()
        savez("http", X_http, y_http)
        # - smtp
        is_smtp = df[2] == 'smtp'
        X_smtp = df[is_smtp].drop(columns=[1, 2, 3, last]).to_numpy()
        y_smtp = (df[is_smtp][last] != 'normal.').to_numpy()
        savez("smtp", X_smtp, y_smtp)
# COVERTYPE
with urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz") as resp:
    with gzip.GzipFile(fileobj=resp) as buffer:
        X, y = parse_uci(buffer, [2], [4], delimiter=",")
        savez("covtype", X, y)

# SATELLITE
with io.StringIO() as buffer:
    with urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn") as trn:
        buffer.write(trn.read().decode("utf-8"))
    with urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.tst") as tst:
        buffer.write(tst.read().decode("utf-8"))
    # - sat1
    buffer.seek(0)
    X, y = parse_uci(buffer, [1, 3, 4, 5, 6, 7], [2])
    savez("sat1", X, y)
    # - sat3
    buffer.seek(0)
    X, y = parse_uci(buffer, [1, 3, 6, 7], [2, 4, 5])
    savez('sat3', X, y)

# SHUTTLE
with io.StringIO() as buffer:
    with urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.trn.Z") as trn:
        buffer.write(unlzw.unlzw(trn.read()).decode("utf-8"))
    with urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.tst") as tst:
        buffer.write(tst.read().decode("utf-8"))
    buffer.seek(0)
    X, y = parse_uci(buffer, [1, 4], [2, 3, 5, 7])
    savez("shuttle", X, y)

# MULCROSS
# with urllib.request.urlopen("https://www.openml.org/data/download/16787460/phpGGVhl9") as resp:
#     with io.StringIO() as buffer:
#         buffer.write(resp.read().decode("utf-8"))
#         buffer.seek(0)
#         data, meta = loadarff(buffer)
#         df = pd.DataFrame(data)
#         X = df.drop(columns=["Target"]).to_numpy()
#         y = (df["Target"] != b'Normal').to_numpy()
#         savez("mulcross", X, y)
