# 加载包
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# %% 数据EDA
df_data_train = pd.read_csv(open(r'./data/train.csv'))
print(df_data_train.describe())
print(df_data_train.info())
df_data_test = pd.read_csv(open(r'./data/test_noLabel.csv'))
##
# Age	员工年龄
# Label	员工是否已经离职，1表示已经离职，2表示未离职
# BusinessTravel	商务差旅频率，Non-Travel不出差，TravelRarely不经常出差，TravelFrequently经常出差
# Department	员工所在部门，Sales销售部，Research & Development研发部，Human Resources人力资源部
# DistanceFromHome	公司跟家庭住址的距离，从1到29，1表示最近，29表示最远
# Education	员工的教育程度，从1到5，5表示教育程度最高
# EducationField	员工所学习的专业领域，Life Sciences表示生命科学，Medical表示医疗，Marketing表示市场营销，
#                   Technical Degree表示技术学位，Human Resources表示人力资源，Other表示其他
# EmployeeNumber	员工号码
# EnvironmentSatisfaction	员工对于工作环境的满意程度，从1到4，1的满意程度最低，4的满意程度最高
# Gender	员工性别，Male表示男性，Female表示女性
# JobInvolvement	员工工作投入度，从1到4，1为投入度最低，4为投入度最高
# JobLevel	职业级别，从1到5，1为最低级别，5为最高级别
# JobRole	工作角色：Sales Executive销售主管，Research Scientist科学研究员，Laboratory Technician实验室技术员，
#                    Manufacturing Director制造总监，Healthcare Representative医疗代表，Manager经理，
#                    Sales Representative销售代表，Research Director研究总监，Human Resources人力资源
# JobSatisfaction	工作满意度，从1到4，1代表满意度最低，4代表最高
# MaritalStatus	员工婚姻状况，Single单身，Married已婚，Divorced离婚
# MonthlyIncome	员工月收入，范围在1009到19999之间
# NumCompaniesWorked	员工曾经工作过的公司数
# Over18	年龄是否超过18岁
# OverTime	是否加班，Yes表示加班，No表示不加班
# PercentSalaryHike	工资提高的百分比
# PerformanceRating	绩效评估
# RelationshipSatisfaction	关系满意度，从1到4，1表示满意度最低，4表示满意度最高
# StandardHours	标准工时
# StockOptionLevel	股票期权水平
# TotalWorkingYears	总工龄
# TrainingTimesLastYear	上一年的培训时长，从0到6，0表示没有培训，6表示培训时间最长
# WorkLifeBalance	工作与生活平衡程度，从1到4，1表示平衡程度最低，4表示平衡程度最高
# YearsAtCompany	在目前公司工作年数
# YearsInCurrentRole	在目前工作职责的工作年数
# YearsSinceLastPromotion	距离上次升职时长
# YearsWithCurrManager	跟目前的管理者共事年数
# %%
lst_content_remove = ['ID', 'EmployeeNumber', 'JobRole', 'JobLevel', 'Gender',
                      'MaritalStatus', 'Department', 'EducationField', 'EnvironmentSatisfaction',
                      'RelationshipSatisfaction', 'YearsWithCurrManager']

for _ in df_data_train:
    lst_content_remove.append(_) if len(df_data_train[_].unique()) == 1 else None
for _ in df_data_test:
    # print(_, len(df_data_test[_].unique()), '' if len(df_data_test[_].unique()) > 20 else df_data_test[_].unique())
    lst_content_remove.append(_) if len(df_data_test[_].unique()) == 1 else None


# df_data_train['MonthlyIncome_p_workyear'] = (df_data_train['MonthlyIncome'] - df_data_train['MonthlyIncome'].min()) / \
#                                             df_data_train['TotalWorkingYears']
# df_data_train['workage'] = df_data_train['Age'] - df_data_train['TotalWorkingYears']
# df_data_train['focus'] = df_data_train['YearsInCurrentRole'] / df_data_train['YearsAtCompany']
# tmp_df = df_data_train[['MonthlyIncome', 'TotalWorkingYears', 'MonthlyIncome_p_workyear', 'Label']].corr()
# tmp_df = df_data_train.corr()

class fe_num_label:
    def __init__(self, dict_fe):
        self.dict_fe = dict_fe

    def get_num_label(self, label):
        return self.dict_fe[label]


def feature_engineering(df):
    df_cp = df.copy()
    #
    df_cp['MonthlyIncome_p_workyear'] = (df_cp['MonthlyIncome'] - df_cp['MonthlyIncome'].min()) / \
                                        (df_cp['TotalWorkingYears'] + 1)  # 平均收入增长=(收入-最小收入)/(工龄+1)
    # df_cp['workage'] = df_cp['Age'] - df_cp['TotalWorkingYears']  # 起始工作年龄=年龄-工龄
    df_cp['focus'] = (df_cp['YearsInCurrentRole'] + 1) / (df_cp['YearsAtCompany'] + 1) #在职时间/总工龄
    fe_d = fe_num_label({'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2})  # 出差情况转化成0,1,2

    df_cp['BusinessTravel'] = df_cp['BusinessTravel'].map(fe_d.get_num_label)
    # Sales销售部，Research & Development研发部，Human
    # Resources人力资源部
    fe_d = fe_num_label({'Life Sciences': 'Research & Development',
                         'Medical': 'Research & Development',
                         'Other': 'Other',
                         'Technical Degree': 'Research & Development',
                         'Human Resources': 'Human Resources',
                         'Marketing': 'Sales'})
    df_cp['department_match'] = (df_cp['EducationField'].map(fe_d.get_num_label) == df_cp['Department']).map(
        lambda x: 1 if x else 0)  # 将学历专业和部门做比对.判断学历和工作内容是否相同,转化成0,1
    df_cp['OverTime'] = df_cp['OverTime'].map(lambda x: 1 if x == 'Yes' else 0)  # 加班转化0,1

    tmp_df = pd.get_dummies(df_cp['JobRole'])  # 工作角色做onehot后,乘以职业级别
    for _ in tmp_df:
        df_cp[_] = tmp_df[_] * df_cp['JobLevel']

    for fe in ['Gender', 'MaritalStatus']:  # 性别和婚姻状态做onehot
        tmp_df = pd.get_dummies(df_cp[fe])
        for _ in tmp_df:
            df_cp[_] = tmp_df[_]

    lst_content_left = [_ for _ in df_cp if _ not in lst_content_remove]
    df_cp = df_cp[sorted(lst_content_left)]
    for _ in df_cp:
        df_cp[_] = df_cp[_].astype(float)
    return df_cp


df_data_train_fe = feature_engineering(df_data_train)
print(df_data_train_fe.corr())
tmp_df = df_data_train_fe.corr()
tmp_col = tmp_df[tmp_df['Label'].map(abs) >= 0].index
df_data_train_fe[[_ for _ in df_data_train_fe if (_ in tmp_col)]].to_csv(r'C:\Users\yi4fi\Desktop\train_data.csv',index=False)
X, Y = df_data_train_fe[[_ for _ in df_data_train_fe if (_ != 'Label') and (_ in tmp_col)]], df_data_train_fe['Label']
# df_data_train_fe, train_y = df_data_train_fe[[_ for _ in df_data_train_fe if (_ != 'Label') and (_ in tmp_col)]], \
#                             df_data_train_fe['Label']
# for _ in range(10):
# train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)
train_x, train_y = X, Y

ss = preprocessing.StandardScaler()  # StandardScaler
train_x = ss.fit_transform(train_x)

# pca = PCA(n_components=0.98)  # 降到2维
# train_x = pca.fit_transform(train_x)

#
test_x = feature_engineering(df_data_test)
test_x[[_ for _ in test_x if (_ in tmp_col)]].to_csv(r'C:\Users\yi4fi\Desktop\test_data.csv',index=False)
test_x = ss.transform(test_x[[_ for _ in test_x if (_ != 'Label') and (_ in tmp_col)]])
# test_x = pca.transform(test_x)
# % 逻辑回归训练

lr = LogisticRegression()
lr.fit(train_x, train_y)
print('-' * 50)
print('训练集:', str(accuracy_score(train_y, lr.predict(train_x))))

    #
    # print('验证集:', str(accuracy_score(test_y, lr.predict(test_x))))


# %%
# def try_different_method(clf):
#     clf.fit(train_x, train_y)
#     score = clf.score(test_x, test_y)
#     # score = clf.score(train_x, train_y)
#     print('the score is :', score)
#
#
# for clf_key in clfs.keys():
#     print('the classifier is :', clf_key)
#     clf = clfs[clf_key]
#     try_different_method(clf)
# %%
df_upload = df_data_test[['ID']]
df_upload['Label'] = lr.predict(test_x)
# #
# clfs['random_forest'].fit(train_x, train_y)
# df_upload['Label'] = clfs['svm'].predict(test_x)
df_upload.to_csv('./data/upload.csv', index=False)
# def ratio(lst):
#     return sum(lst)/len(lst)
#
# tmp_df = df_data_train.corr()