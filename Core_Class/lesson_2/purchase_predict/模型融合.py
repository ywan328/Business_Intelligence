from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
import pandas as pd
# 导入精度和召回
from sklearn.metrics import precision_score, recall_score, roc_auc_score

path = '../客户购买预测/'
LGBM = pd.read_csv(path+'output/baseline_lightGBM.csv')
XGB = pd.read_csv(path+'output/baseline_XGBoost.csv')
CATB = pd.read_csv(path+'output/baseline_CatBoost.csv')


# 加权平均 模型融合
pre = LGBM[['ID']]
pre['pred'] = (LGBM['pred'] + XGB['pred'] + CATB['pred']) / 3
pre.to_csv(path+'output/baseline_融合.csv', index=False)



