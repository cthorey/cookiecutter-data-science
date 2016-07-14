
from src.model.base import *
import xgboost as xgb
import numpy as np


class ObjectiveXGboost(object):

    ''' 
    custom = True. Prediction from the model are softmax.
    If not, so not.

    '''

    def softmax(self, score):
        score = np.asarray(score, dtype=float)
        score -= np.max(score)
        score = np.exp(score)
        score /= np.sum(score, axis=1)[:, np.newaxis]
        return score

    def loss_brier(self, preds, labels, w):
        ''' Given a N*C preds and labels matrix, return grad,hess '''

        # Loss
        N = preds.shape[0]
        loss = (1. / N) * np.sum((preds - labels)**2 * w)

        # Grads
        tmp = np.sum(preds * (preds - labels) * w, axis=1)[:, np.newaxis]
        grads = 2 * preds * ((preds - labels) * w - tmp)
        grads = grads.flatten()

        # Hess
        line0 = 2 * preds * (1 - preds) * ((preds - labels) * w - tmp)
        line1 = 2 * preds * preds * w * (1 - preds)
        tmphess = np.sum(w * (2 * preds - labels), axis=1)[:, np.newaxis]
        line2 = 2 * preds * preds * \
            (w * (2 * preds - labels) - preds * tmphess)
        hess = line0 + line1 - line2
        hess = hess.flatten()

        return loss, grads, hess

    def loss_softmax(self, preds, labels):
        # Loss
        N = preds.shape[0]
        loss = - (1. / float(N)) * np.sum(labels * np.log(preds))

        # Grads
        tmp = np.sum(labels, axis=1)[:, np.newaxis]
        grads = -labels + preds * tmp
        grads = grads.flatten()

        # Hess
        hess = preds * (1 - preds) * tmp
        hess = hess.flatten()
        return loss, grads, hess

    def loss_weighted_softmax(self, preds, labels, w):
        # Loss
        N = preds.shape[0]
        loss = - (1. / float(N)) * np.sum(labels * np.log(preds))

        # Grads
        tmp = np.sum(labels * w, axis=1)[:, np.newaxis]
        grads = -labels * w + preds * tmp
        grads = grads.flatten()

        # Hess
        hess = preds * (1 - preds) * tmp
        hess = hess.flatten()
        return loss, grads, hess

    def format_output(self, preds, dX, custom=True):
        ''' Given N*C matrix.
        For CV, cannot use directly the given pro matrix. Short cut instead.
        '''
        if custom:
            preds = self.softmax(preds)
        else:
            # If objective == 'multi:sofprob ou softmax,pas besoin de softmax
            # result
            pass
        try:
            labels = np.array(getattr(dX, 'y_given'))
        except:
            N = preds.shape[0]
            labels = np.zeros_like(preds)
            y_given = dX.get_label().astype('int')
            labels[range(N), y_given] = 1

        assert preds.shape == labels.shape, 'Shape preds {}, label {}'.format(
            preds.shape, labels.shape)
        return preds, labels

    def custom_obj(self, preds, dX, method, custom=True, w=None):
        preds, labels = self.format_output(preds, dX, custom)
        if w != None:
            _, grad, hess = getattr(self, method)(preds, labels, w)
        else:
            _, grad, hess = getattr(self, method)(preds, labels)
        return grad, hess

    def custom_err(self, preds, dX, method, custom=True, w=None):
        preds, labels = self.format_output(preds, dX, custom)
        if w != None:
            loss, _, _ = getattr(self, method)(preds, labels, w)
        else:
            _, grad, hess = getattr(self, method)(preds, labels)
        return method, loss

    def brier_obj(self, preds, dX):
        return self.custom_obj(preds, dX, method='loss_brier', w=self.weights)

    def brier_error(self, preds, dX):
        return self.custom_err(preds, dX, method='loss_brier', w=self.weights, custom=False)

    def brier_error_custom(self, preds, dX):
        return self.custom_err(preds, dX, method='loss_brier', w=self.weights)

    def softmax_obj(self, preds, dX):
        return self.custom_obj(preds, dX, method='loss_softmax')

    def softmax_error_custom(self, preds, dX):
        return self.custom_err(preds, dX, method='loss_softmax')

    def w_softmax_obj(self, preds, dX):
        return self.custom_obj(preds, dX, method='loss_weighted_softmax', w=self.weights)

    def w_softmax_error_custom(self, preds, dX):
        return self.custom_err(preds, dX, method='loss_weighted_softmax', w=self.weights)


class BaseXGBoostExperiment(BaseExperiment, ObjectiveXGboost):
    '''Handle a classic XGBoostExperiment'''

    def setup_model_parameters(self):
        # Xgboost
        self.typemodel = 'xgboost'
        # CV

        if not hasattr(self, 'num_boost_round'):
            self.num_boost_round = 20
        if not hasattr(self, 'nfold'):
            self.nfold = 3
        params = dict(objective='multi:softmax',
                      num_class=len(self.classes),
                      subsample=0.9,
                      min_child_weight=1,
                      gamma=0.2,
                      colsample_bytre=0.65,
                      max_depth=5,
                      eta=0.15)
        self.cv_args = dict(params=params,
                            num_boost_round=self.num_boost_round,
                            early_stopping_rounds=10,
                            verbose_eval=10,
                            nfold=self.nfold,
                            stratified=True,
                            metrics=['mlogloss'])

        # train
        params = dict(objective='multi:softprob',
                      num_class=len(self.classes),
                      subsample=0.9,
                      max_depth=4,
                      eta=0.1,
                      colsample_bytree=0.8)
        self.train_args = dict(params=params,
                               num_boost_round=20,
                               early_stopping_rounds=10,
                               verbose_eval=10)

        self.resume = False

    def train_cv(self, dtrain, **args):
        cv_args = self.parse_args(args, method='cv')
        print(cv_args)
        cv_xgb = xgb.cv(dtrain=dtrain,
                        **cv_args)
        results = self.persit_result_cv(cv_xgb, args)
        return results

    def train_model(self, dtrain, dtest, **args):
        '''Train and save idx model. idx refer to the rank of the model. idx=0 retru
        the best model '''

        train_args = self.parse_args(args, method='train')
        print(train_args)
        watchlist = [(dtrain, 'train'), (dtest, 'eval')]
        return xgb.train(dtrain=dtrain,
                         evals=watchlist,
                         **train_args)

    def train_k_model(self, dtrain, dtest, idx=0, **args):
        '''Train and save idx model. idx refer to the rank of the model. idx=0 retrun
        the best model '''

        self.modelname = self.results().iloc[idx].fname
        print('Using {} cv params to produce the model'.format(self.modelname))
        kwargs = self.load(name=os.path.join(self.dirname,
                                             '{}.json'.format(self.modelname)))

        kwargs.update(args)
        train_args = self.parse_args(kwargs, method='train')
        print('Params {}'.format(train_args))
        watchlist = [(dtrain, 'train'), (dtest, 'eval')]
        self.model = xgb.train(dtrain=dtrain,
                               evals=watchlist,
                               **train_args)
        self.model_args = train_args

        model_file = os.path.join(
            self.dirname, '{}.model'.format(self.modelname))
        self.model.save_model(model_file)
        best_attr = {f: v for f, v in self.model.__dict__.items() if f.split('_')[
            0] == 'best'}
        self.dump(best_attr, model_file + '_bestattr')

        return self.model

    def load_model(self, idx=0):
        self.modelname = self.results().iloc[idx].fname
        print('Load model {}'.format(self.modelname))
        model_file = os.path.join(self.dirname,
                                  '{}.model'.format(self.modelname))
        if not os.path.isfile(model_file):
            raise ValueError(
                'First, train the model {}!!'.format(self.modelname))

        kwargs = self.load(name=os.path.join(self.dirname,
                                             '{}.json'.format(self.modelname)))
        train_args = self.parse_args(kwargs, method='train')

        self.model = xgb.Booster(model_file=model_file)
        self.model_args = train_args
        print(self.load('{}_bestattr'.format(model_file)))
        for key, val in self.load('{}_bestattr'.format(model_file)).items():
            setattr(self.model, key, val)

        return self.model

    def parse_args(self, kwargs, method):
        '''help process inputed parameters.
        method is either train or cv
        '''
        assert all([len(f.split('__')) == 2 for f in kwargs]), (
            'Be carefull, you fool. Kwargs argument have to follow params__blabla or args__blabla')
        assert all([f.split('__')[0] in ['params', 'args'] for f in kwargs]), (
            'Be carefull, you fool. Kwargs argument have to follow params__blabla or args__blabla')

        params = {k.split(
            '__')[-1]: v for k, v in kwargs.items() if k.split('__')[0] == 'params'}
        main_args = {k.split(
            '__')[-1]: v for k, v in kwargs.items() if k.split('__')[0] == 'args'}

        params = self.control_type(params)
        main_args = self.control_type(main_args)

        args = dict(getattr(self, '{}_args'.format(method)))
        args['params'].update(params)
        args.update(main_args)
        self.check_args(args)

        return args

    def check_args(self, args):

        # Exeption handling
        if 'feval' in args.keys():
            if args['feval'].split('_')[-1] == 'custom' and 'obj' not in args.keys():
                raise ValueError(
                    'If obj is not passed. feval have to be the non custom version !!')
            if args['feval'].split('_')[-1] != 'custom' and 'obj' in args.keys():
                raise ValueError(
                    'If obj is passed. feval have to be the custom version (softamx preds!!)')

            if not hasattr(self, args['feval']):
                raise ValueError('{} not implemented'.format(args['feval']))
            else:
                args['feval'] = getattr(self, args['feval'])

        if 'obj' in args.keys():
            if args['params']['objective'] != 'reg:linear':
                raise ValueError(
                    'Objective have to be reg:linear for custom obj !')
                if 'feval' in args.keys():
                    if (args['params']['objective'] != 'multi:softprob') and ('obj' not in args.keys()):
                        raise ValueError(
                            'objective have to be set to multi:softprob or design (custom) if feval is passed. Current:{}'.format(args['params']['objective']))
            if not hasattr(self, args['obj']):
                raise ValueError('{} not implemented'.format(args['obj']))
            else:
                args['obj'] = getattr(self, args['obj'])

    def persit_result_cv(self, cv_xgb, args):

        results = cv_xgb.iloc[-1].to_dict()
        loss = [f for f in results.keys() if f.split('-')[0] == 'test' and
                f.split('-')[-1] == 'mean'][-1]
        results['loss'] = results[loss]

        # persist data
        metric = '-'.join(loss.split('-')[1:-1])
        fname = 'cvtest_{:.5}_cvtrain_{:.5}'.format(
            results['test-{}-mean'.format(metric)], results['train-{}-mean'.format(metric)])
        csv_name = os.path.join(
            self.dirname, '{}.csv'.format(fname))
        json_name = os.path.join(
            self.dirname, '{}.json'.format(fname))
        cv_xgb.to_csv(csv_name, index=False)
        self.dump(obj=args, name=json_name)
        return results

    def results(self):
        models = [f for f in os.listdir(self.dirname) if f.endswith('csv')]
        dfmodels = pd.DataFrame(
            [pd.read_csv(os.path.join(self.dirname, f)).iloc[-1].to_dict() for f in models])
        metric = '-'.join(dfmodels.columns[0].split('-')[1:-1])
        dfmodels['score'] = dfmodels['test-{}-mean'.format(metric)]
        dfmodels['fname'] = [os.path.splitext(f)[0] for f in models]

        return dfmodels.sort_values('score')

    def results_params(self):

        dfmodels = self.results()
        params = [os.path.join(self.dirname, f) for f in
                  os.listdir(self.dirname) if f.endswith('json') and
                  f not in ['experiment.json']]
        params_name = self.load(params[0]).keys()
        dfparams = pd.DataFrame([self.load(f)for f in params])
        df = dfmodels.join(dfparams)
        return df.sort_values('score')

    def predict(self, dX):
        ''' BE CAREFULL WHAT IF THE OBJ. IF it is reg:linear.
        self.model.predict return a score that u need to softmax
        before getting probabilities ! '''

        print('Make prediction with model {}'.format(self.modelname))
        if hasattr(self.model, 'best_ntree_limit'):
            # Needed due to the API of xgboost
            y_pred = self.model.predict(
                dX, ntree_limit=self.model.best_ntree_limit)
        else:
            y_pred = self.model.predict(dX)

        if self.model_args['params']['objective'] == 'reg:linear':
            y_pred = self.softmax(y_pred)

        return y_pred

    def control_type(self, args):
        ''' return the good type for the parameters'''
        dict_type = {}
        dict_type.update(
            {f: int for f in ['max_depth',
                              'min_child_weight',
                              'num_class',
                              'nfold',
                              'early_stopping_rounds',
                              'verbose_eval',
                              'num_boost_round',
                              'scale_pos_weight']})
        dict_type.update(
            {f: float for f in ['learning_rate',
                                'gamma',
                                'reg_alpha',
                                'colsample_bytree',
                                'subsample',
                                'eta']})
        dict_type.update(
            {f: str for f in ['objective', 'obj', 'feval']})

        dict_type.update(
            {f: list for f in ['eval_metric', 'metrics']})
        return {key: dict_type[key](val) for key, val in args.items()}
