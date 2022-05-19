import os
import numpy as np
import os.path
import random

#read = lambda p: np.loadtxt(p, delimiter="\t")[::5, :7]
read = lambda p: np.loadtxt(p, delimiter="\t")[::5, :14]

def load_dataset(file_path):
    paths_list = []
    paths_len = []

    for p in os.listdir(file_path):
        fname = os.path.join(file_path, p)
        path = read(fname)
        paths_list.append(path)
        paths_len.append(len(path))

    max_length = np.max(paths_len)
    N = len(paths_list)

    paths = [np.pad(p, [(0, max_length - len(p)), (0, 0)]) for p in paths_list]

    dataset = []
    targets = []
    for i in range(N):
        for j in range(paths_len[i] - 1):
            data = np.concatenate([paths[i][j], paths[i][paths_len[i] - 1]], axis=-1)

            targets.append(paths[i][j + 1])
            dataset.append(data)

    data = list(zip(dataset, targets))
    random.shuffle(data)
    dataset, targets = zip(*data)
    return np.asarray(dataset).astype(np.float32), np.asarray(targets).astype(np.float32)

def load_test_dataset(file_path):
    paths_list = []
    paths_len = []

    for p in os.listdir(file_path):
        fname = os.path.join(file_path, p)
        path = read(fname)
        paths_list.append(path)
        paths_len.append(len(path))

    max_length = np.max(paths_len)
    N = len(paths_list)

    paths = [np.pad(p, [(0, max_length - len(p)), (0, 0)]).astype(np.float32) for p in paths_list]

    return paths, paths_len
