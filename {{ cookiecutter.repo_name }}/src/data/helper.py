import os
import sys
sys.path.append(os.environ['ROOT_DIR'])
from setting import *
from src.visualization.visualize import Sequence
import pandas as pd
import json
import numpy as np


class BaseTransformer(object):

    def update(self, x):
        self.data = self.data.append(self.transform(x), ignore_index=True)


def load_meta(name):
    return json.load(open(os.path.join(RAW, '{}.json'.format(name))))


def load_sequence(file_id, loader_features, split):
    filename = str(file_id).zfill(5)
    sequence = Sequence(RAW, os.path.join(RAW, split, '{}').format(filename))
    tmp = loader_features(sequence, split)
    tmp['record_id'] = file_id
    tmp['start'] = [f for f in range(int(sequence.meta['end']))]
    tmp['end'] = [f for f in range(1, int(sequence.meta['end']) + 1)]
    return tmp


def load_sequences(file_ids, loader_features, split='train'):
    control_file_ids(file_ids, split)
    df = pd.DataFrame()
    for file_id in file_ids:
        print('*' * 50)
        print('Processing of the {} sequence'.format(file_id))
        print('*' * 50)
        tmp = load_sequence(file_id, loader_features, split)
        df = df.append(tmp)
    return df


def control_file_ids(file_ids, split):
    correct = get_split_sequences(split)
    test = [f for f in file_ids if f in correct]
    assert len(test) == len(
        file_ids), ("sequences have to be a subset of {}: {}".format(split, correct))


def get_split_sequences(split):
    fids = [f for f in map(int, os.listdir(os.path.join(RAW, split)))]
    return fids


def save_features(df, file_name):
    df.to_csv('{}.csv'.format(file_name), index=False)

# ACC FEATURE EXTRACTOR


def MAV(x):
    return np.abs(x).mean()


def WMAV_w1(x):
    N = len(x)
    N_max = int(0.75 * N)
    N_min = int(0.25 * N)
    w1 = np.ones(N) * 0.5
    w1[N_min:N_max] = float(1)
    return w1


def WMAV_w2(x):
    N = len(x)
    N_max = int(0.75 * N)
    N_min = int(0.25 * N)
    w2 = np.ones(N)
    w2_idx = np.array([f for f in range(N)])
    w2[:N_min] = 4 * w2_idx[:N_min] / float(N)
    w2[N_max:] = 4 * (N - w2_idx[N_max:]) / float(N)
    return w2


def WMAV1(x):
    w1 = WMAV_w1(x)
    return (np.abs(x) * w1).mean()


def WMAV2(x):
    w2 = WMAV_w2(x)
    return (np.abs(x) * w2).mean()


def HM(x):
    N = len(x)
    if N == 0:
        return np.nan
    else:
        return float(N) / np.sum(1 / x)


def RMS(x):
    return np.sqrt(np.mean(x**2))


def CL(x):
    N = len(x)
    return np.sum(np.abs(x[1:N] - x[:N - 1]))


def ZC(x):
    return len(np.where(np.diff(np.sign(x)))[0])


def SSI(x):
    return np.sum(x**2)
