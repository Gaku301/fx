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

# 説明変数となる行列Xを作成
day_ago = 25
num_sihyou = 1

X = np.zeros((len(data2), day_ago*num_sihyou))

for i in range(0, day_ago):
    X[i:len(data2),i] = data2[0:len(data2)-i,4]

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

# 学習データを使用して、線形回帰モデルを作成
linear_reg_model = linear_model.LinearRegression()

linear_reg_model.fit(X_train, Y_train)


# 2017年のデータで予測し、グラフで予測具合を確認
Y_pred = linear_reg_model.predict(X_test)
print(Y_pred)

result = pd.DataFrame(Y_pred)
result.columns = ['Y_pred']
result['Y_test'] = Y_test
sns.set_style('darkgrid')
sns.regplot(x='Y_pred', y='Y_test', data=result)
# plt.show()

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
plt.show()
