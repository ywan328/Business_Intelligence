import os
import json
import gc
from numba import jit
from tqdm import tqdm_notebook
from tqdm import tqdm

# Integrated model
import lightgbm as lgb
import catboost as cbt
import xgboost as xgb

# base import
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# about sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
# about time
import time
import datetime
from datetime import datetime, timedelta

# Garbage collection
import gc
# scipy
from scipy.signal import hilbert
from scipy.signal import hanning
from scipy.signal import convolve
from scipy import stats
import scipy.spatial.distance as dist
# other
from collections import Counter
from statistics import mode
# warning
import warnings

warnings.filterwarnings("ignore")
import json
import math
from itertools import product
import ast
from sklearn.model_selection import train_test_split  # 数据分隔出训练集和验证集
# 导入精度和召回
from sklearn.metrics import precision_score, recall_score

# TPOP AutoML
from tpot import TPOTClassifier

from os.path import dirname, join
current_dir = dirname(__file__)

pd.set_option('display.max_columns', None)


class ModelSetup(object):
    default = {
        'lgb': {'boosting_type': 'gbdt',
                'num_leaves': 31,
                'reg_alpha': 10,
                'reg_lambda': 5,
                'max_depth': -1,
                'n_estimators': 500,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'subsample_freq': 2,
                'min_child_samples': 10,
                'learning_rate': 0.05,
                'random_state': 2019,
                'early_stopping_rounds': 100,
                'eval_metric': 'mae',
                'verbose': 100
                },

        'ctb': {'iterations': 1000,
                'depth': 5,
                'learning_rate': 0.1,
                'loss_function': 'RMSE',
                'early_stopping_rounds': 100,
                'eval_metric': 'mae',
                'verbose': 100
                },

        'xgb': {'max_depth': 8,
                'learning_rate': 0.1,
                'n_estimators': 160,
                'silent': False,
                'early_stopping_rounds': 50,
                'eval_metric': 'mae',
                'verbose': True
                },
        'xgb_c': {'max_depth': 8,
                  'learning_rate': 0.2,
                  'n_estimators': 160,
                  'silent': False,
                  'early_stopping_rounds': 50,
                  'eval_metric': 'mae',
                  'verbose': True,
                  'min_child_weight':10,
                  'nthread':1,
                  'subsample':0.8
                },
        'lr': {},
        'id3_c': {'criterion': 'entropy'},
        'ada_c': {'n_estimators': 100,
                  'learning_rate':1.0,
                  'algorithm': 'SAMME.R'
                  },
    }

    def __init__(self, model_setup=None):
        if model_setup is None:  # load default setting
            self.items = ModelSetup.default
        elif type(model_setup) is str:  # load from json file
            self.load(model_setup)

    def set_values(self, model_name, values):
        self.items[model_name] = ModelSetup.SplitParameter(values)

    def set(self, model_name, key, value):
        self.items[model_name][key] = value

    def load(self, path):
        raise NotImplementedError()

    def save(self, path):
        raise NotImplementedError()

    @staticmethod
    def SplitParameter(values):
        # TODO: value type must be check and converted properly.
        setup = dict()
        values = values.split(',')
        for paar in values:
            paar = paar.strip()
            paar = paar.split('=')
            value = paar[1].strip()
            setup[paar[0].strip()] = int(value) if value.isdigit() else float(value)
        return setup

class ModelFactory(object):
    def __init__(self, model_type, random_state=30, model_setup=None):
        self.model_type = model_type

        self.modelSetup = model_setup if type(model_setup) is ModelSetup else ModelSetup(model_setup)

        setup = self.modelSetup.items[model_type]

        #TODO: property should be setup dynamic!!
        if model_type == 'lgb':
            self.model = lgb.LGBMRegressor(boosting_type=setup['boosting_type'],
                                           num_leaves=setup['num_leaves'],
                                           reg_alpha=setup['reg_alpha'],
                                           reg_lambda=setup['reg_lambda'],
                                           max_depth=setup['max_depth'],
                                           n_estimators=setup['n_estimators'],
                                           subsample=setup['subsample'],
                                           colsample_bytree=setup['colsample_bytree'],
                                           subsample_freq=setup['subsample_freq'],
                                           min_child_samples=setup['min_child_samples'],
                                           learning_rate=setup['learning_rate'])
        elif model_type == 'ctb':
            self.model = cbt.CatBoostRegressor(iterations=setup['iterations'],
                                               depth=setup['depth'],
                                               learning_rate=setup['learning_rate'],
                                               loss_function=setup['loss_function'])
        elif model_type == 'xgb':
            self.model = xgb.XGBRegressor(max_depth=setup['max_depth'],
                                          learning_rate=setup['learning_rate'],
                                          n_estimators=setup['n_estimators'],
                                          silent=setup['silent'])  # objective='reg:gamma'
        elif model_type == 'xgb_c':
            self.model = xgb.XGBClassifier(max_depth=setup['max_depth'],
                                          learning_rate=setup['learning_rate'],
                                          min_child_weight=setup['min_child_weight'],
                                          n_estimators=setup['n_estimators'],
                                          nthread=setup['nthread'],
                                          subsample=setup['subsample'])
        elif model_type == 'lr':
            self.model = LogisticRegression()
        elif model_type == 'id3_c':
            self.model = DecisionTreeClassifier(criterion=setup['criterion'])
        elif model_type == 'ada_c':
            self.model = AdaBoostClassifier(n_estimators=setup['n_estimators'])

        self.model.random_state = random_state

    #TODO: Setup for InstanceOf  and calling should be seperated.
    def fit(self, train_x, train_y, eval_set, sample_weight=None, categorical_features=None):
        setup = self.modelSetup.items[self.model_type]
        self.model.random_state = self.model.random_state + 1
        if self.model_type == 'lgb':
            try:
                self.model.fit(train_x,
                               train_y,
                               eval_set=eval_set,
                               categorical_feature=categorical_features,
                               sample_weight=sample_weight,
                               early_stopping_rounds=setup['early_stopping_rounds'],
                               eval_metric=setup['eval_metric'],
                               verbose=setup['verbose'])
            except:
                self.model.fit(train_x,
                               train_y,
                               eval_set=eval_set,
                               # categorical_feature=self.cate_feature,
                               sample_weight=sample_weight,
                               early_stopping_rounds=setup['early_stopping_rounds'],
                               eval_metric=setup['eval_metric'],
                               verbose=setup['verbose'])
        elif self.model_type == 'ctb':
            self.model.fit(train_x,
                           train_y,
                           eval_set=eval_set,
                           sample_weight=sample_weight,
                           early_stopping_rounds=setup['early_stopping_rounds'],
                           # eval_metric=setup['eval_metric'],categorical_feature=categorical_features,
                           verbose=setup['verbose'])
        elif self.model_type == 'xgb':
            self.model.fit(train_x,
                           train_y,
                           eval_set=eval_set,
                           sample_weight=sample_weight,
                           early_stopping_rounds=setup['early_stopping_rounds'],
                           eval_metric=setup['eval_metric'],
                           verbose=setup['verbose'])
        elif self.model_type == 'xgb_c':
            self.model.fit(train_x, train_y)
        elif self.model_type == 'lr':
            self.model.fit(train_x, train_y)
        elif self.model_type == 'id3_c':
            self.model.fit(train_x, train_y)
        elif self.model_type == 'ada_c':
            self.model.fit(train_x, train_y)

    def predict(self, data):
        return self.model.predict(data)

    def predict_prob(self, data):
        return self.model.predict_proba(data)

    def get_importances(self):
        return self.model.feature_importances_


class ModelResult(object):
    def __init__(self, ids):
        self.result = dict()
        self.ids = ids

    def add_result(self, model_name, params, alias):
        raise NotImplementedError()

    def get_result(self, model_name, params):
        raise NotImplementedError()

    def get_result_with_alias(self, alias):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()


class MLearning(object):

    def __init__(self):
        self.data = None
        self.train = None
        self.test = None
        self.test_pre = None
        self.label = ''
        self.features = []
        self.unusedFeatures = []
        self.categorical_features = []
        self.numeric_features = []
        self.feature_importances = pd.DataFrame()

    def read_train(self, path, enc='gbk'):
        try:
            self.train = pd.read_csv(path, encoding=enc)
        except:
            self.train = pd.read_csv(join(current_dir, path), encoding=enc)

        self.print_train_shape()

    def read_test(self, path, enc='gbk'):
        try:
            self.test = pd.read_csv(path, encoding=enc)
        except:
            self.test = pd.read_csv(join(current_dir, path), encoding=enc)
        self.print_test_shape()

    def train_corr_graph(self):
        MLearning.display_corr(self.train.corr())

    def test_corr_graph(self):
        MLearning.display_corr(self.test.corr())

    def corr_graph(self):
        MLearning.display_corr(self.data.corr())

    def print_train_shape(self):
        print("The shape of train set：{}".format(self.train.shape))

    def print_test_shape(self):
        print("The shape of test set：{}".format(self.test.shape))

    def stat_train_empty(self):
        return self.train.isnull().sum()

    def stat_test_empty(self):
        return self.test.isnull().sum()

    def set_features_without(self, useless=[]):
        self.unusedFeatures = useless
        self.features = [i for i in self.data.columns if i not in useless]
        print("Features：{}".format(len(self.features)))
        return self.features

    def set_categorical_features(self, categorical_features):
        self.categorical_features = categorical_features

    def add_label_for_test(self, label, default=-1):
        self.label = label
        self.test[label] = default

    def merge_train_and_test(self):
        self.data = self.train.append(self.test).reset_index(drop=True)

    def drop_column(self, columns):
        self.data.drop(columns, axis=1, inplace=True)

    def fill_none(self, col_name, default=0):
        self.data[col_name].fillna(default, inplace=True)

    def fill_none_for_more(self, col_names, default=0):
        for col in col_names:
            self.fill_none(col, default)

    def convert_datetime(self, datetime_col_name, translate_prefix=''):
        self.data[datetime_col_name] = pd.to_datetime(self.data[datetime_col_name], format='%Y-%m-%d %H:%M:%S')
        if translate_prefix is not None:
            self.data[translate_prefix + "weekday"] = self.data[datetime_col_name].dt.weekday + 1
            self.data[translate_prefix + "year"] = self.data[datetime_col_name].dt.year
            self.data[translate_prefix + "quarter"] = self.data[datetime_col_name].dt.quarter
            self.data[translate_prefix + "hour"] = self.data[datetime_col_name].dt.hour
            self.data[translate_prefix + "month"] = self.data[datetime_col_name].dt.month

    def convert_day_in_year(self, month_col_name, day_col_name, target_name):
        self.data[target_name] = self.data.apply(lambda r: MLearning.day_in_year(r['month'], r['day']), axis=1)

    def add_workday(self, workday_col_name, weekday_col_name):
        self.data[workday_col_name] = self.data[weekday_col_name].apply(MLearning.get_workday)

    def add_hour_cut(self, hour_cut_col_name, hour_col_name):
        self.data[hour_cut_col_name] = self.data[hour_col_name].apply(MLearning.hour_cut)

    def label_encoder_without(self, useless=[]):
        cat_col1 = [i for i in self.data.select_dtypes(object).columns if i not in useless]
        for i in tqdm_notebook(cat_col1):
            self.data[i] = LabelEncoder().fit_transform(self.data[i].astype(str))

    def label_encoder(self, used=None):
        cat_col1 = [i for i in self.data.select_dtypes(object).columns if (used is None) or (i in used)]
        for i in tqdm_notebook(cat_col1):
            self.data[i] = LabelEncoder().fit_transform(self.data[i].astype(str))

    def one_hot(self, columns=[]):
        for col in columns:
            tmp = pd.get_dummies(self.data[col], prefix=col)
            self.data = self.data.drop([col], axis=1)
            self.data = pd.concat([self.data, tmp], axis=1)

    def min_max_scaler(self, columns=[]):
        scaler = MinMaxScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])

    def z_score(self, columns=[]):
        for col in columns:
            self.data[col] = (self.data[col] - self.data[col].mean()) / self.data[col].std()

    def save_predict(self, col_id, predict_label, path):
        if type(self.test_pre) is pd.Series:
            self.test_pre = self.test_pre.values

        pd.DataFrame({col_id: self.test[col_id], predict_label: self.test_pre})\
            .to_csv(join(current_dir, path), index=False)

    def n_fold_predict(self, model_type, model_setup=None, random_state=2018, n_splits=5, shuffle=True):
        if 'sample_weight' not in self.data.keys():
            self.data['sample_weight'] = 1

        self.feature_importances['feature'] = self.features

        # generate model
        model = ModelFactory(model_type, random_state, model_setup)

        predict_label = 'predict_' + self.label
        self.data[predict_label] = 0

        test_index = (self.data[self.label].isnull()) | (self.data[self.label] == -1)
        test_data = self.data[test_index]
        train_data = self.data[~test_index].reset_index(drop=True)

        fold = 0
        k_fold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for train_idx, val_idx in k_fold.split(train_data):
            train_x = train_data.loc[train_idx][self.features]
            train_y = train_data.loc[train_idx][self.label]
            test_x = train_data.loc[val_idx][self.features]
            test_y = train_data.loc[val_idx][self.label]
            sample_weight = train_data.loc[train_idx]['sample_weight']

            model.fit(train_x, train_y, [(test_x, test_y)], sample_weight, self.categorical_features)

            train_data.loc[val_idx, predict_label] = model.predict(test_x)
            if len(test_data) != 0:
                test_data[predict_label] = test_data[predict_label] + model.predict(test_data[self.features])

            self.feature_importances['fold_{}'.format(fold + 1)] = model.get_importances()
            fold+=1
        test_data[predict_label] = test_data[predict_label] / n_splits

        self.test_pre = test_data[predict_label]

        return self.test_pre

    def predict(self, model_type, model_setup=None, random_state=100, test_size=0.3, shuffle=True):
        x = self.data[self.data[self.label] >= 0][self.features]
        y = self.data[self.data[self.label] >= 0][self.label]
        testx = self.data[self.data[self.label] < 0][self.features]

        train_x, vali_x, train_y, vali_y = train_test_split(x, y, test_size=test_size, random_state=random_state)
        random_state += 1

        model = ModelFactory(model_type, random_state, model_setup)

        model.fit(train_x, train_y, [(train_x, train_y), (vali_x, vali_y)])

        vali_pre = model.predict(vali_x)
        print('LR AUC:           ' + str(roc_auc_score(vali_y, vali_pre)))

        self.feature_importances['feature'] = self.features
        self.feature_importances['important'] = model.get_importances()

        self.test_pre = model.predict(testx)
        return self.test_pre

    def run_tpot(self, path, random_state=40, test_size=0.3):
        x = self.data[self.data[self.label] >= 0][self.features]
        y = self.data[self.data[self.label] >= 0][self.label]
        train_x, vali_x, train_y, vali_y = train_test_split(x, y, test_size=test_size, random_state=random_state)
        tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
        tpot.fit(train_x, train_y)
        print(tpot.score(vali_x, vali_y))
        tpot.export(join(current_dir, path))

    @staticmethod
    def display_corr(corr):
        # 查看特征直接的相关性
        # get_ipython().run_line_magic('matplotlib', 'inline')
        sns.set(style="white")
        # 设置 matplotlib f的尺寸
        plt.subplots(figsize=(11, 9)) #f, ax =
        # 生成自定义的散色图
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        # 为热力图设置长宽比
        sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.show()

    @staticmethod
    def get_workday(x):
        if 0 < x < 6:
            return 1
        else:
            return 0

    @staticmethod
    def hour_cut(x):
        if 0 <= x < 6:
            return 0
        elif 6 <= x < 8:
            return 1
        elif 8 <= x < 12:
            return 2
        elif 12 <= x < 14:
            return 3
        elif 14 <= x < 18:
            return 4
        elif 18 <= x < 21:
            return 5
        elif 21 <= x < 24:
            return 6

    @staticmethod
    # 出生的年代
    def birth_split(x):
        if 1920 <= x <= 1930:
            return 0
        elif 1930 < x <= 1940:
            return 1
        elif 1940 < x <= 1950:
            return 2
        elif 1950 < x <= 1960:
            return 3
        elif 1960 < x <= 1970:
            return 4
        elif 1970 < x <= 1980:
            return 5
        elif 1980 < x <= 1990:
            return 6
        elif 1990 < x <= 2000:
            return 7

    Month_To_Nr = [('jan', 31),
                   ('feb', 28),
                   ('mar', 31),
                   ('apr', 30),
                   ('may', 31),
                   ('jun', 30),
                   ('jul', 31),
                   ('aug', 31),
                   ('sep', 30),
                   ('oct', 31),
                   ('nov', 30),
                   ('dec', 31)]
    @staticmethod
    # 当年的第几天
    def day_in_year(month, day, reverse = True):
        count = 0
        for m in MLearning.Month_To_Nr:
            if month[0:3].lower() == m[0]:
                if day < 1 or day > m[1]:
                    raise ValueError('day:%d'%day)
                break
            else:
                count += m[1]
        return 365 - (count + day) if reverse else count + day

    @staticmethod
    # 收入分组
    def income_cut(x):
        if x < 0:
            return 0
        elif 0 <= x < 1200:
            return 1
        elif 1200 < x <= 10000:
            return 2
        elif 10000 < x < 24000:
            return 3
        elif 24000 < x < 40000:
            return 4
        elif 40000 <= x:
            return 5

    @staticmethod
    def reduce_mem_usage(df, columns=None, verbose=True):
        # 通过检测列的最小值与最大值，来使用最合适的data type, 以减小内存的使用
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024 ** 2
        for col in df.columns if columns is None else columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024 ** 2
        if verbose:
            print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                    start_mem - end_mem) / start_mem))
        return df
