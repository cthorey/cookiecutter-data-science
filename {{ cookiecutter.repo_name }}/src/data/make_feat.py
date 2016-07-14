import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pandas as pd
from tqdm import *
from functools import reduce
import argparse
import scipy.stats as st
import statsmodels.api as sm

# add the 'src' directory as one where we can import modules
sys.path.append(os.environ['ROOT_DIR'])
from setting import *

from src.visualization.visualize import Sequence
from src.data.helper import *

# Prior features


class PIRRawTransformer(BaseTransformer):
    ''' PIR transform '''

    def __init__(self):
        self.functions = {'mean': np.mean,
                          'std': np.std,
                          'min': np.min,
                          'med': np.median,
                          'max': np.max}
        self.rooms = load_meta('pir_locations')
        cols = [['pir_{}_{}'.format(ff, room) for room in self.rooms]
                for ff in self.functions.keys()]
        self.cols = reduce(lambda x, y: x + y, cols)
        self.data = pd.DataFrame(columns=self.cols)

    def transform(self, x):
        tmp0 = {'pir_{}'.format(key): x.apply(ff, axis=0).to_dict()
                for key, ff in self.functions.items()}
        tmp1 = {}
        for key, dic in tmp0.items():
            tmp1.update({'{}_{}'.format(key, subkey): subval
                         for subkey, subval in dic.items()})

        return tmp1


class RSSIRawTransformer(BaseTransformer):
    ''' RSSI transform '''

    def __init__(self):
        self.functions = {'mean': np.mean,
                          'std': np.std,
                          'min': np.min,
                          'med': np.median,
                          'max': np.max}
        self.rooms = load_meta('access_point_names')
        cols = [['rssi_{}_{}'.format(ff, room) for room in self.rooms]
                for ff in self.functions.keys()]
        self.cols = reduce(lambda x, y: x + y, cols)
        self.data = pd.DataFrame(columns=self.cols)

    def transform(self, x):
        tmp0 = {'rssi_{}'.format(key): x.fillna(float(-115)).apply(ff, axis=0).to_dict()
                for key, ff in self.functions.items()}
        tmp1 = {}
        for key, dic in tmp0.items():
            tmp1.update({'{}_{}'.format(key, subkey): subval
                         for subkey, subval in dic.items()})

        return tmp1


class ACCRawTransformer(BaseTransformer):
    ''' Acceleration transform '''

    def __init__(self):
        self.functions = {'mean': np.mean,
                          'std': np.std,
                          'min': np.min,
                          'med': np.median,
                          'max': np.max}
        self.axes = load_meta('accelerometer_axes')
        cols = [['acc_{}_{}'.format(ff, axe) for axe in self.axes]
                for ff in self.functions.keys()]
        self.cols = reduce(lambda x, y: x + y, cols)
        self.data = pd.DataFrame(columns=self.cols)

    def transform(self, x):
        tmp0 = {'acc_{}'.format(key): x.apply(ff, axis=0).to_dict()
                for key, ff in self.functions.items()}
        tmp1 = {}
        for key, dic in tmp0.items():
            tmp1.update({'{}_{}'.format(key, subkey): subval
                         for subkey, subval in dic.items()})

        return tmp1


class ACCRawTransformer_Improved(BaseTransformer):
    '''
    Accelration transforme based on
    http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130851
    '''

    def __init__(self):
        self.functions = dict(mav=MAV,
                              hm=HM,
                              var=np.var,
                              rms=RMS,
                              skewness=st.skew,
                              kurtosis=st.kurtosis,
                              cl=CL,
                              zc=ZC,
                              ssi=SSI)
        self.axes = load_meta('accelerometer_axes')
        cols = [['acc_{}_{}'.format(ff, axe) for axe in self.axes]
                for ff in self.functions.keys()]
        self.cols = reduce(lambda x, y: x + y, cols)
        self.cols_name = self.cols + ['acc_scc_x', 'acc_scc_y', 'acc_scc_z',
                                      'acc_cor_xy', 'acc_cor_xz', 'acc_cor_yz',
                                      'acc_cor_xx', 'acc_cor_yy', 'acc_cor_zz']
        self.data = pd.DataFrame(columns=self.cols)

    def corr(self, x, y):
        try:
            return np.corrcoef(x, y)[1, 0]
        except:
            return np.nan

    def get_correlation(self, x):
        corr = dict(acc_cor_xy=self.corr(x['x'], x['y']),
                    acc_cor_xz=self.corr(x['x'], x['z']),
                    acc_cor_yz=self.corr(x['y'], x['z']),
                    acc_cor_xx=self.corr(x['x'], x['x']),
                    acc_cor_yy=self.corr(x['y'], x['y']),
                    acc_cor_zz=self.corr(x['z'], x['z']))
        return corr

    def get_ssc(self, x):
        '''slope sign change'''
        accx = np.array(x['x'])
        accy = np.array(x['y'])
        accz = np.array(x['z'])
        t = np.array(x.index)
        scc = dict(acc_scc_x=len(np.where(np.diff(np.sign((accx[1:] - accx[:-1]) / (t[1:] - t[:-1]))))[0]),
                   acc_scc_y=len(
                       np.where(np.diff(np.sign((accy[1:] - accy[:-1]) / (t[1:] - t[:-1]))))[0]),
                   acc_scc_z=len(np.where(np.diff(np.sign((accz[1:] - accz[:-1]) / (t[1:] - t[:-1]))))[0]))
        return scc

    def transform(self, x):
        tmp0 = {'acc_{}'.format(key): x.apply(ff, axis=0).to_dict()
                for key, ff in self.functions.items()}
        tmp1 = {}
        for key, dic in tmp0.items():
            tmp1.update({'{}_{}'.format(key, subkey): subval
                         for subkey, subval in dic.items()})

        corr = self.get_correlation(x)
        tmp1.update(corr)

        scc = self.get_ssc(x)
        tmp1.update(scc)

        return tmp1


class VIDEORawTransformer(BaseTransformer):
    ''' Acceleration transform '''

    def __init__(self, location='k'):
        self.location = location
        self.functions = {'mean': np.mean,
                          'std': np.std,
                          'min': np.min,
                          'med': np.median,
                          'max': np.max}
        self.features = reduce(
            lambda x, y: x + y, list(load_meta('video_feature_names').values()))
        cols = [['video_{}_{}_{}'.format(self.location, ff, feat) for feat in self.features]
                for ff in self.functions.keys()]
        self.cols = reduce(lambda x, y: x + y, cols)
        self.data = pd.DataFrame(columns=self.cols)

    def transform(self, x):
        tmp0 = {'video_{}_{}'.format(self.location, key): x.apply(ff, axis=0).to_dict()
                for key, ff in self.functions.items()}
        tmp1 = {}
        for key, dic in tmp0.items():
            tmp1.update({'{}_{}'.format(key, subkey): subval
                         for subkey, subval in dic.items()})

        return tmp1


class ROOMRawTransformer(BaseTransformer):

    data = pd.DataFrame(columns=['room'])

    def transform(self, x):
        if len(x.name.values) == 0:
            data = {'room': np.nan}
        else:
            data = {'room': x.name.values[0]}
        return data


def load_features(sequence, split='train'):
    if split not in ['train', 'test']:
        raise ValueError('split has to be either train/test.')

    pir_collector = PIRRawTransformer()
    rssi_collector = RSSIRawTransformer()
    acc_collector = ACCRawTransformer()
    acc_collector_improved = ACCRawTransformer_Improved()
    video_collector_lr = VIDEORawTransformer('lr')
    video_collector_k = VIDEORawTransformer('k')
    video_collector_h = VIDEORawTransformer('h')
    if split == 'train':
        room_collector = ROOMRawTransformer()

    size = int(sequence.meta['end'])
    for _, (acc, rssi, pir, vid_lr, vid_k, vid_h), loc in tqdm(sequence.iterate(split),
                                                               total=size):
        pir_collector.update(pir)
        rssi_collector.update(rssi)
        acc_collector.update(acc)
        acc_collector_improved.update(acc)
        video_collector_lr.update(vid_lr)
        video_collector_k.update(vid_k)
        video_collector_h.update(vid_h)
        if split == 'train':
            room_collector.update(loc)

    # reduce step
    features = pir_collector.data
    for collector in [rssi_collector, acc_collector, acc_collector_improved,
                      video_collector_h, video_collector_k,
                      video_collector_lr]:
        features = features.join(collector.data)
    if split == 'train':
        feats = json.load(open(os.path.join(RAW, 'annotations.json')))
        features = features.join(sequence.targets[feats].iloc[:-1])
        features = features.join(room_collector.data)

    return features

# Rooms feature
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, help='name store file')

    args = vars(parser.parse_args())
    filename = os.path.join(PROCESSED, 'processed_{}'.format(args['split']))
    file_ids = get_split_sequences(split=args['split'])
    df = load_sequences(file_ids=file_ids,
                        loader_features=load_features,
                        split=args['split'])
    save_features(df, filename)
