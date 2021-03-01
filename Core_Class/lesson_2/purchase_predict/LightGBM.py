from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
import pandas as pd
# 导入精度和召回
from sklearn.metrics import precision_score, recall_score, roc_auc_score

path = '../客户购买预测/'
train = pd.read_csv(path+'input/train_set.csv')
test = pd.read_csv(path+'input/test_set.csv')

# 将训练集和测试集合并在一起
test['y'] = -1
data = train.append(test).reset_index(drop = True)

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler as std
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import f1_score

# 只选出数据格式是字符串（object）的特征
cat_col = [i for i in data.select_dtypes(object).columns if i not in ['ID','y']]
for i in cat_col:
    lbl = LabelEncoder()
    data[i] = lbl.fit_transform(data[i].astype(str))

feats = [i for i in data.columns if i not in ['ID','y']]
tar = data[data['y']!=-1][feats]
y = data[data['y']!=-1]['y']
test = data[data['y']==-1][feats]

#5折交叉验证
from sklearn.model_selection import KFold,StratifiedKFold
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42) #shuffle就是打乱训练集

model = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=30, reg_alpha=0, reg_lambda=0.,
    max_depth=-1, n_estimators=1500, objective='binary', metric='auc',
    subsample=0.95, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.02, random_state=2017
)

pre = data[data['y']==-1][['ID']]
pre['pred'] = 0
oof = np.zeros(len(tar),)

for train_idx, vali_idx in kfold.split(tar):
    model.random_state = model.random_state + 1
    train_x = tar.loc[train_idx]
    train_y = y.loc[train_idx]
    val_x = tar.loc[vali_idx]
    val_y = y.loc[vali_idx]
    model.fit(
        train_x, train_y,
        eval_set=[(train_x, train_y), (val_x, val_y)],
        categorical_feature=cat_col,
        eval_metric='logloss',
        early_stopping_rounds=10,
        verbose=1
    )
    oof[vali_idx] = model.predict_proba(val_x)[:, 1]
    pre['pred'] += model.predict_proba(test)[:, 1]

pre['pred'] = pre['pred']/n_splits
# 线下验证
print('线下AUC：', roc_auc_score(y, oof))
pre.to_csv(path+'output/baseline_lightGBM.csv', index=False)






