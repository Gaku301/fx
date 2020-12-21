import numpy as np  
import pandas as pd
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns
from sklearn import linear_model

import warnings
warnings.filterwarnings('ignore')


# dataフォルダのパス
data_dir = './data/'

# fxデーターの読み込み
data = pd.read_csv(data_dir + "USDJPY_1997_2017.csv")

data2 = np.array(data)

# # 説明変数となる行列Xを作成
# day_ago = 25
# num_sihyou = 1 + 4 + 4 +4

# X = np.zeros((len(data2), day_ago*num_sihyou))

# for s in range(0, num_sihyou):
#     for i in range(0, day_ago):
#         X[i:len(data2),day_ago*s+i] = data2[0:len(data2)-i,s+4]

# # 被説明変数となる Y = pre_day後の終値-当日終値を作成
# Y = np.zeros(len(data2))

# # 何日後を値段の差を予測するのか決める
# pre_day = 1
# Y[0:len(Y)-pre_day] = X[pre_day:len(X),0] - X[0:len(X)-pre_day,0]

# # X,Yの正規化
# original_X = np.copy(X)
# tmp_mean = np.zeros(len(X))

# for i in range(day_ago,len(X)):
#     tmp_mean[i] = np.mean(original_X[i-day_ago+1:i+1,0])
#     for j in range(0, X.shape[1]):
#         X[i,j] = (X[i,j] - tmp_mean[i])
#     Y[i] = Y[i]

# # XとYを学習データとテストデータ(2017年〜)に分ける
# X_train = X[200:5193,:]
# Y_train = Y[200:5193]

# X_test = X[5193:len(X)-pre_day,:]
# Y_test = Y[5193:len(Y)-pre_day]

# # 学習データを使用して、線形回帰モデルを作成
# linear_reg_model = linear_model.LinearRegression()

# linear_reg_model.fit(X_train, Y_train)


# # 2017年のデータで予測し、グラフで予測具合を確認
# Y_pred = linear_reg_model.predict(X_test)

# result = pd.DataFrame(Y_pred)
# result.columns = ['Y_pred']
# result['Y_test'] = Y_test

# sns.set_style('darkgrid')
# sns.regplot(x='Y_pred', y='Y_test', data=result)
# # plt.show()

# # 正答率の計算
# success_num = 0
# for i in range(len(Y_pred)):
#     if Y_pred[i] * Y_test[i] >=0:
#         success_num+=1

# print("予測日数:"+ str(len(Y_pred))+"、正答日数:"+str(success_num)+"、正答率:"+str(success_num/len(Y_pred)*100))



# # -------------------------------------------
# # 2017年の予測結果の合計を計算
# # 前々日終値に比べて前日終値が高い場合は、買い
# sum_2017 = 0

# for i in range(0,len(Y_test)):
#     if Y_pred[i] >= 0:
#         sum_2017 += Y_test[i]
#     else:
#         sum_2017 -= Y_test[i]

# # print("2017年の利益合計:%1.3lf" %sum_2017)

# # 予測結果の総和グラフ
# total_return = np.zeros(len(Y_test))

# if Y_pred[i] >=0:
#     total_return[0] = Y_test[i]
# else:
#     total_return[0] = -Y_test[i]

# for i in range(1, len(result)):
#     if Y_pred[i] >=0:
#         total_return[i] = total_return[i-1] + Y_test[i]
#     else:
#         total_return[i] = total_return[i-1] - Y_test[i]

# plt.plot(total_return)
# plt.show()

# 5日移動平均を追加
data2 = np.c_[data2, np.zeros((len(data2),1))]
ave_day = 5
for i in range(ave_day, len(data2)):
    tmp = data2[i-ave_day+1:i+1,4].astype(np.float)
    data2[i,5] = np.mean(tmp)

# 25日移動平均を追加
data2 = np.c_[data2, np.zeros((len(data2),1))]
ave_day = 25
for i in range(ave_day, len(data2)):
    tmp = data2[i-ave_day+1:i+1,4].astype(np.float)
    data2[i,6] = np.mean(tmp)

# 75日移動平均を追加
data2 = np.c_[data2, np.zeros((len(data2),1))]
ave_day = 75
for i in range(ave_day, len(data2)):
    tmp = data2[i-ave_day+1:i+1,4].astype(np.float)
    data2[i,7] = np.mean(tmp)

# 200日移動平均を追加
data2 = np.c_[data2, np.zeros((len(data2),1))]
ave_day = 200
for i in range(ave_day, len(data2)):
    tmp = data2[i-ave_day+1:i+1,4].astype(np.float)
    data2[i,8] = np.mean(tmp)


# 一目均衡表を追加(9,26,52)
para1 = 9
para2 = 26
para3 = 52

# 転換線 = (過去(para1)日間の高値 + 安値) / 2
data2 = np.c_[data2, np.zeros((len(data2),1))]
for i in range(para1, len(data2)):
    tmp_high = data2[i-para1+1:i+1,2].astype(np.float)
    tmp_low = data2[i-para1+1:i+1,3].astype(np.float)
    data2[i,9] = (np.max(tmp_high) + np.min(tmp_low)) / 2

# 基準線 = (過去(para2)日間の高値 + 安値) / 2
data2 = np.c_[data2, np.zeros((len(data2),1))]
for i in range(para2, len(data2)):
    tmp_high = data2[i-para2+1:i+1,2].astype(np.float)
    tmp_low = data2[i-para2+1:i+1,3].astype(np.float)
    data2[i,10] = (np.max(tmp_high) + np.min(tmp_low)) / 2

# 先行スパン1 = {(転換値 + 基準値) / 2}を(para2)日先にずらしたもの
data2 = np.c_[data2, np.zeros((len(data2),1))]
for i in range(0, len(data2)-para2):
    tmp = (data2[i,9] + data2[i,10]) / 2
    data2[i+para2,11] = tmp
    
# 先行スパン2 = {(過去(para3)日間の高値 + 安値) / 2}を(para2)日先にずらしたもの
data2 = np.c_[data2, np.zeros((len(data2),1))]
for i in range(para3,len(data2)-para2):
    tmp_high = data2[i-para3+1:i+1,2].astype(np.float)
    tmp_low = data2[i-para3+1:i+1,3].astype(np.float)
    data2[i+para2,12] = (np.max(tmp_high) + np.min(tmp_low)) / 2


# 25日ボリンジャーバンド(±1, 2シグマ)を追加
parab = 25
data2 = np.c_[data2, np.zeros((len(data2),4))]
for i in range(parab, len(data2)):
    tmp = data2[i-parab+1:i+1,4].astype(np.float)
    data2[i,13] = np.mean(tmp) + 1.0* np.std(tmp)
    data2[i,14] = np.mean(tmp) - 1.0* np.std(tmp)
    data2[i,15] = np.mean(tmp) + 2.0* np.std(tmp)
    data2[i,15] = np.mean(tmp) - 2.0* np.std(tmp)


# 説明変数となる行列Xを作成
day_ago = 25
num_sihyou = 1 + 4 + 4 + 4

X = np.zeros((len(data2), day_ago*num_sihyou))

for s in range(0, num_sihyou):
    for i in range(0, day_ago):
        X[i:len(data2),day_ago*s+i] = data2[0:len(data2)-i,s+4]

# 被説明変数となる Y = pre_day後の終値-当日終値を作成
Y = np.zeros(len(data2))

# 何日後を値段の差を予測するのか決める
pre_day = 1
Y[0:len(Y)-pre_day] = X[pre_day:len(X),0] - X[0:len(X)-pre_day,0]

# X,Yの正規化
original_X = np.copy(X)
tmp_mean = np.zeros(len(X))

for i in range(day_ago,len(X)):
    tmp_mean[i] = np.mean(original_X[i-day_ago+1:i+1,0])
    for j in range(0, X.shape[1]):
        X[i,j] = (X[i,j] - tmp_mean[i])
    Y[i] = Y[i]

# XとYを学習データとテストデータ(2017年〜)に分ける
X_train = X[200:5193,:]
Y_train = Y[200:5193]

X_test = X[5193:len(X)-pre_day,:]
Y_test = Y[5193:len(Y)-pre_day]

linear_reg_model = linear_model.LinearRegression()

linear_reg_model.fit(X_train, Y_train)

# 2017年のデータで予測し、グラフで予測具合を確認
Y_pred = linear_reg_model.predict(X_test)
# print(Y_pred)

result = pd.DataFrame(Y_pred)
result.columns = ['Y_pred']
result['Y_test'] = Y_test

sns.set_style('darkgrid')
sns.regplot(x='Y_pred', y='Y_test', data=result)
plt.show()

# 正答率の計算
success_num = 0
for i in range(len(Y_pred)):
    if Y_pred[i] * Y_test[i] >=0:
        success_num+=1

# print("予測日数:"+ str(len(Y_pred))+"、正答日数:"+str(success_num)+"、正答率:"+str(success_num/len(Y_pred)*100))


# -------------------------------------------
# 2017年の予測結果の合計を計算
# 前々日終値に比べて前日終値が高い場合は、買い
sum_2017 = 0

for i in range(0,len(Y_test)):
    if Y_pred[i] >= 0:
        sum_2017 += Y_test[i]
    else:
        sum_2017 -= Y_test[i]

print("2017年の利益合計:%1.3lf" %sum_2017)

# 予測結果の総和グラフ
total_return = np.zeros(len(Y_test))

if Y_pred[i] >=0:
    total_return[0] = Y_test[i]
else:
    total_return[0] = -Y_test[i]

for i in range(1, len(result)):
    if Y_pred[i] >=0:
        total_return[i] = total_return[i-1] + Y_test[i]
    else:
        total_return[i] = total_return[i-1] - Y_test[i]

plt.plot(total_return)

