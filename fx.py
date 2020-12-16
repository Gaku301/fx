import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# dataフォルダのパス
data_dir = "./data/"

# fxデータの読み込み
data = pd.read_csv(data_dir + "USDJPY_1997_2017.csv")

# Close-Openをデータに追加
data['Change'] = data.Close - data.Open

# 2016年のデータ
data16 = data.iloc[4935:5193,:]

# 2017年のデータ
data17 = data.iloc[5193:,:]


# 2016年のデータを計算
sum_2016 = 0
for i in range(2,len(data16)):
    if data16.iloc[i-2,4] <= data16.iloc[i-1,4]:
        sum_2016 += data16.iloc[i,5]
    else:
        sum_2016 -= data16.iloc[i,5]


# 2017年のデータを計算
sum_2017 = 0
for i in range(2,len(data17)):
    if data17.iloc[i-2,4] <= data17.iloc[i-1,4]:
        sum_2017 += data17.iloc[i,5]
    else:
        sum_2017 -= data17.iloc[i,5]


# print("2017年の利益合計:%1.3lf" %sum_2017)

plt.style.use('seaborn-darkgrid')
plt.plot(data17['Close'])
plt.ylim([95,125])
plt.show()

