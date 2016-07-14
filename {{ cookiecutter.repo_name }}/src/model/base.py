"""
BASE STRUCTURE FOR COMPETITION.

class BaseDataPipeline to handle the
transfo from processed data to X_train,y_train,X_val,y_val,X_test,y_test

class BaseModel to handle neccesary step for the model

class BaseExperiment to handle the experiment.
"""
import pandas as pd
import os
import sys
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
sys.path.append(os.environ['ROOT_DIR'])

from collections import OrderedDict
from time import strftime
import itertools
import pickle
from random import shuffle

# import src.model.base_pipeline as FeatureTransformer

# See at the end of the file for an exemple of
# a class in hooks_pipeline
# from src.model.hooks_pipeline import *


class BaseDataPipeline(object):

    '''Helper to make reusable pipe to transform the data '''

    def build_one_pipeline(self, pipe_list, pipe_kwargs):
        ''' Given a pipe_list and pipe_kwargs, return
        one pipeline '''
        pipe = ""
        return pipe

    def build_entire_pipeline(self, features, shared_steps):
        '''
        Build a pipeline base on first
        FeatureUnion to build the features
        Indexer to return a dataframe with the good
        index and the necessary columns
        '''
        # Features
        pipe_list = features['pipe_list']
        pipe_kwargs = features['pipe_kwargs']
        try:
            assert pipe_list.keys() == pipe_kwargs.keys()
            features = FeatureUnion([(key, self.build_one_pipeline(
                pipe_list[key], pipe_kwargs[key])) for key in pipe_list.keys()])
        except:
            raise ValueError('You pipe shared_steps list is fucked up')

        # Shared steps
        if len(shared_steps) == 0:
            pipeline = Pipeline([
                ('features', features)
            ])
        else:
            pipe_list = shared_steps['pipe_list']
            pipe_kwargs = shared_steps['pipe_kwargs']
            try:
                shared_steps = self.build_one_pipeline(pipe_list, pipe_kwargs)
            except:
                raise ValueError('Your pipe common list is fucked up')

            pipeline = Pipeline([
                ('features', features),
                ('shared_steps', shared_steps)
            ])

        return pipeline

    def dump_pipeline(self):
        ''' Dump the fitted pipeline for reusability '''

        pipeline = self.build_entire_pipeline(**self.pipe_def)
        data = pd.read_csv(self.f_train_data)
        train, test = self.train_test_split(data)
        pipeline.fit(train)
        with open(os.path.join(self.dirname, 'pipeline.pickle'), 'wb') as f:
            pickle.dump(pipeline, f)

    def load_pipeline(self):
        with open(os.path.join(self.dirname, 'pipeline.pickle'), 'rb') as f:
            pipeline = pickle.load(f)
        return pipeline

    def setup_data_parameters(self):
        '''
        setup the path to raw/processed data +
        meta info
        '''
        pass

    def setup_pipe_def(self):
        '''
        Often, processed data is a huge thing with evertyghin inside.
        A pipeline is needed to extract what we want.

        Help for reusability. 

        Both features and shared_steps have to behave as:

        dict(pipe_list=pipe_list, pipe_kwargs=pipe_kwargs)

        PLEASE UUSE ORDEREDDICT TO PREVENT ANY PROBLEM WHEN
        RELOADING THE WHOLE THING. ITERATION over pipe_list.kesy().
        THIS SHOULD NOT CHANGE.

        Below is an example
        '''

        # FeatureSelector is a class that has to be defined somewhere
        pipe_list = {'feature0': ['FeatureSelector']}
        pipe_kwargs = {'feature0': {'FeatureSelector__cols': 'feature0'}}
        features = OrderedDict(pipe_list=pipe_list, pipe_kwargs=pipe_kwargs)

        # shared_steps defined a second form o transformation which is now
        # apply over all transformed features such as PCA
        pipe_list = {'step0': ['PCA']}
        pipe_kwargs = {'step0': {'PCA__n_components': 10}}

        shared_steps = OrderedDict(
            pipe_list=pipe_list, pipe_kwargs=pipe_kwargs)
        self.pipe_def = OrderedDict(
            features=features, shared_steps=shared_steps)

    def train_test_split(self, data):
        '''
        Raw processing on data + splitting + reindexing

        ATENTION: I HAD VERY BAD PB IN THE PAST BECAUSE I FORGOT
        TO REINDEX THE DATAFRAME. IN PARTICULAR, JOIN OPERATION CAN
        BE PROBLEMATIC. DO IT !!!        

        '''
        train = ''
        test = ''
        return train, test

    def load_training_data(self, verbose=0):
        '''return train,test,optional labelencoder'''
        data = pd.read_csv(self.f_train_data)
        train, test = self.train_test_split(data)
        pipeline = self.load_pipeline()
        X_train = pipeline.transform(train)
        X_test = pipeline.transform(test)

        labelencoder = LabelEncoder()
        labelencoder.fit(train.room)
        y_train = labelencoder.transform(train.room)
        y_test = labelencoder.transform(test.room)

        if verbose > 0:
            print ('All training data finite:', np.all(np.isfinite(X_train)))
            print ('All training data finite:', np.all(np.isfinite(y_train)))
            print ('All training data finite:', np.all(np.isfinite(X_test)))
            print ('All training data finite:', np.all(np.isfinite(y_test)))
            """
            Print simple statistics regarding the number of instances
            """
            print ("Training data shapes:")
            print ("train_x.shape: {}".format(X_train.shape))
            print ("train_y.shape: {}".format(y_train.shape))
            print

            print ("Testing data shapes")
            print ("test_x.shape: {}".format(X_test.shape))
            print ("test_y.shape: {}".format(y_test.shape))

            # Last thing, the label has to be a one column vector.

            # Format for xgboost
        dtrain = xgb.DMatrix(X_train, y_train)
        dtest = xgb.DMatrix(X_test, y_test)

        pass

    def load_test_data(self):
        '''for submission'''
        pass


class BaseModel(object):
    ''' Base class to handle model behavior'''

    def setup_model_parameters(self):
        ''' Setup the model parameters.
        in one big dictionary.
        '''

        self.typemodel = ''
        self.cv_args = {}
        self.train_args = {}
        self.resume = False

    def train_cv(self, X_train, y_train, **args):
        ''' train the model using cross_val
        X_train : training data
        args is a dict with all the parameters necessary
        to stuff into the model.

        If several level are needed use __ to separate them.
        '''
        cv_args = self.parse_args(args, method='train')
        pass

    def persist_result_cv(self, cv_result, args):
        ''' Here it is important to store the results.

        This method should create two files:
        cvtest_{:.5}_cvtrain_{:.5}.csv : A pd dataframe
        which store the result.
        cvtest_{:.5}_cvtrain_{:.5}.json: A json file which store
        args for reproducibility.
        '''

        cvtest = 1
        cvtrain = 1
        csv_name = 'cvtest_{:.5}_cvtrain_{:.5}.csv'.format(cvtest, cvtrain)
        json_name = 'cvtest_{:.5}_cvtrain_{:.5}.json'.format(cvtest, cvtrain)
        cv_result.to_csv(csv_name, index=False)
        self.dump(obj=args, name=json_name)

    def results(self):
        ''' Return a dataframe with the result for each model run in the
        experiement as well as the parameters '''

        models = [f for f in os.listdir(self.dirname) if f.endswith('csv')]
        dfmodels = pd.DataFrame()
        dfmodels['fname'] = [os.path.splitext(f)[0] for f in models]
        params = [os.path.join(self.dirname, f) for f in
                  os.listdir(self.dirname) if f.endswith('json') and
                  f not in ['experiment.json']]
        dfmodels['score'] = 1

        params_name = self.load(params[0]).keys()
        params = [list(map(lambda x: x[-1] if type(x) == list else x,
                           self.load(f).values())) for f in params]

        dfparams = pd.DataFrame(params, columns=params_name)
        df = dfmodels.join(dfparams)
        return df.sort_values('score')

    def run_cross_validation(self, dtrain, grid_params, args):
        self.grid_params = grid_params
        cv_results = {}
        generator = self.brut_exp_generator()
        for i, run in enumerate(self.brut_exp_generator()):
            args.update(run)
            res = self.train_cv(dtrain, **args)
            res.update(run)
            cv_results['exp_{}'.format(i)] = res
        df = pd.DataFrame([f for f in cv_results.values()])
        metrics = [f for f in df.columns if f.split(
            '-')[0] == 'test' and f.split('-')[-1] == 'mean'][0]
        return df.sort_values(metrics)

    def train_model(self, X_train, y_train, **args):
        ''' train the model without cv
        X_train : training data
        args is a dict with all the parameters necessary
        to stuff into the model.

        If several level are needed use __ to separate them.
        '''
        train_args = self.parse_args(args, method='train')
        pass

    def parse_args(self, kwargs, method):
        '''
        Usefull to pars the args given to the model to
        give them the proper shape or type.
        '''
        assert all([len(f.split('__')) == 2 for f in kwargs]), (
            'Be carefull, you fool. Kwargs argument have to follow params__blabla or args__blabla')
        assert all([f.split('__')[0] in ['params', 'args'] for f in kwargs]), (
            'Be carefull, you fool. Kwargs argument have to follow params__blabla or args__blabla')

        self.check_type()
        self.check_args(args)
        return

    def check_type():
        ''' Check the type of the params.

        Usefull with oscar !'''
        pass

    def check_args(args):
        '''handle expection'''
        pass

    def train_k_model(self, dtrain, dtest, idx=0):
        '''
        train the K ieme best model in self.dirname
        and save the model as well as the parameters of this
        model
        '''

        self.model = ''
        self.model_args = ''

    def load_model(self, idx=0):
        self.modelname = self.results().iloc[idx].fname
        model_file = os.path.join(self.dirname,
                                  '{}.model'.format(self.modelname))

        if not os.path.isfile(model_file):
            raise ValueError(
                'First, train the model {}!!'.format(self.modelname))

        kwargs = self.load(name=os.path.join(self.dirname,
                                             '{}.json'.format(self.modelname)))
        train_args = self.parse_args(kwargs, method='train')
        #self.model = InstanceModel(file=model_file)
        self.model = ""
        self.model_args = train_args
        return self.model, self.model_args

    def predict(self, dX):
        print('Make prediction with model {}'.format(self.modelname))
        y_pred = self.model.predict(dX)
        return y_pred

    def make_submission(self, dtest):
        pass


class BaseExperiment(BaseDataPipeline, BaseModel):
    ''' Base experiment class to handle the run of a special model

    Required parameter in the experiment hash table:

    - parent_dirname: Parent directory
    - expname: Name of this specific experiment to store the experiment
    - resume: Either to resume a specific expe or not

    method:
         init_worktree: init the worktree
         dump: dump the experiment to json
         load: load a previosuly existing json file
    ...
    '''

    def __init__(self, parent_dirname, expname, **kwargs):
        self.expname = expname
        self.parent_dirname = parent_dirname
        self.dirname = os.path.join(self.parent_dirname, self.expname)
        if 'resume' not in kwargs.keys():
            self.resume = False
        else:
            self.resume = kwargs['resume']
        if self.resume:
            self.load_experiment('experiment.json')
        else:
            self.setup_experiment(**kwargs)
            # Make sure we start we the same thing when we
            # load the experiment in future runs. If notn
            # might be problem with dict order. For instance
            # build pipeline can result in different column order
            # and it is a mess
            self.load_experiment('experiment.json')

    def init_worktree(self):
        '''Init worktree to store models '''
        try:
            os.mkdir(self.dirname)
        except:
            raise ValueError(
                'The directory already exist. Remove it manually or use resume=True')

    def dump(self, obj, name):
        fname = os.path.join(self.dirname, name)
        with open(fname, 'w+') as f:
            json.dump(obj, f,
                      sort_keys=True,
                      indent=4,
                      ensure_ascii=False)

    def load(self, name):
        with open(name, 'r') as f:
            obj = json.load(f, object_pairs_hook=OrderedDict)
        return obj

    def setup_experiment(self, **kwargs):
        # init_worktree
        self.init_worktree()

        # setup data Pipeline to process it from raw
        self.setup_pipe_def()

        # setupe_data, for reproducibility
        self.setup_data_parameters()

        # dump fitted pipeline
        self.dump_pipeline()

        # Overwrite with passed parameters
        for key, val in kwargs.items():
            setattr(self, key, val)

        # setup_model, for reproducibilty
        self.setup_model_parameters()

        # Setup exploration mode
        self.setup_exploration_mode()

        # dump everything
        self.dump(obj=self.__dict__, name='experiment.json')

    def load_experiment(self, name):
        fexpe = os.path.join(self.dirname, name)
        experiment = self.load(name=fexpe)
        for key, val in experiment.items():
            setattr(self, key, val)

    def setup_exploration_mode(self):
        pass

    def get_current_datetime(self):
        return strftime('%Y%m%d_%H%M%S')

    def brut_exp_generator(self):

        if not hasattr(self, 'grid_params'):
            raise ValueError('Must pass a grid_params to brut search')

        # adjust the format
        for key, p in self.grid_params.items():
            if type(p) == dict:
                self.grid_params[key] = [
                    f for f in range(p['min'], p['max'], 1)]

        # create the grid
        product = [x for x in itertools.product(*self.grid_params.values())]
        runs = [dict(zip(self.grid_params.keys(), p)) for p in product]
        shuffle(runs)
        for run in runs:
            yield run


class BaseEnsemble(object):
    ''' handle ensemble'''

    def __init__(self, experiments, parent_dir, resume):
        ''' 
        models = [(experiment,k),...]
        experiment : experiment to considers
        k is the numbrer of model to use from this exp
        '''
        self.experiments = experiments
        self.resume = resume
        self.ensname = self.get_ens_name(experiments)

        self.dirname = os.path.join(parent_dir, self.ensname)
        if not resume:
            self.init_worktree()

    def get_ens_name(self, experiments):
        tupls = [(e.expname, k) for e, k in experiments]
        tupls.sort()
        return '-'.join(['{}_{}'.format(name, k) for name, k in tupls])

    def init_worktree(self):
        '''Init worktree to store models '''
        try:
            os.mkdir(self.dirname)
        except:
            raise ValueError(
                'The directory already exist. Remove it manually or use resume=True')

    def predict(self):
        ''' return an array of prediction
        for the training/validation set '''

        ptrain = []
        pval = []
        ptest = []
        for exp, k in self.experiments:
            print('-' * 50)
            print('Exp {} with {} models'.format(exp.expname, k))
            make_pred = getattr(self, 'predict_{}'.format(exp.typemodel))
            ptrain_tmp, pval_tmp, ptest_tmp = make_pred(
                exp, k)
            ptrain += ptrain_tmp
            pval += pval_tmp
            ptest += ptest_tmp
        preds = np.array(ptrain), np.array(pval), np.array(ptest)
        self.dump_preds(preds)
        return preds

    def dump_preds(self, preds):
        pred_file = os.path.join(self.dirname, 'preds.picke')
        with open(pred_file, 'wb') as f:
            pickle.dump(preds, f)

    def load_preds(self):
        pred_file = os.path.join(self.dirname, 'preds.picke')
        if not os.path.isfile(pred_file):
            raise ValueError('Run the prediciton method first')

        with open(pred_file, 'rb') as f:
            preds = pickle.load(f)
        return preds

    def score_ensemble(self, method='mean'):
        '''
        Score the ensemble according to
        a metrics which has to be defined down
        the line
        '''
        ptrain, ptest, _ = self.load_preds()
        if method == 'mean':
            ptrain = np.mean(ptrain, axis=0)
            ptest = np.mean(ptest, axis=0)
        else:
            raise ValueError('Method not Impemented: {}'.format(method))
        y_train, y_test = self.get_labels()
        score_train = self.metrics(y_train, ptrain)
        score_test = self.metrics(y_test, ptest)
        print('Train: {}'.format(score_train))
        print('Test: {}'.format(score_test))
        return score_train, score_test

    def metrics(self, y_train, ptrain):
        score = 0
        return score

    def get_labels(self):
        return y_train, y_test

    def make_submission(self):
        pass


#  These two class would be better in a hooks_pipeline.py file
class FeatureSelector(TransformerMixin, BaseEstimator):

    def __init__(self, cols=[]):
        self.cols = cols

    def transform(self, X, **transform_params):
        tmp = X[self.cols]
        return tmp

    def fit(self, X, y=None, **fit_params):
        return self


class Inputer(TransformerMixin, BaseEstimator):

    def __init__(self, fill=0):
        self.fill = fill

    def transform(self, X, **transform_params):
        return X.fillna(self.fill)

    def fit(self, X, y=None, **fit_params):
        return self
