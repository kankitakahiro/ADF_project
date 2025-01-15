import numpy as np
import sys, os
from collections import Counter
sys.path.append("../")
from adf_data.census import census_data
from adf_data.credit import credit_data
from adf_data.bank import bank_data
from adf_utils.config import census, credit, bank

'''
データセットからそれぞれの属性の値になにが入っていたのかを調査する関数を作成する
'''

class dataset_config:
    # クラスで最初に実行される
    def __init__(self, dataset):
        self.dataset = dataset
        self.data = {"census": census_data, "credit": credit_data, "bank": bank_data}
        self.data_config = {"census": census, "credit": credit, "bank": bank}
    
    # datasetの属性値を調査する
    def anlz_dataset(self):
        X, Y, input_shape, nb_classes = self.data[self.dataset]()

        # 列ごとの出現頻度を計算
        summary = self.summarize_column_counts(X)

        # 属性名を取得してソート
        data_attr = sorted(summary.keys(), key=lambda x: float(x))  # 数字の順番に並べ替え

        # 属性ごとの値を収集
        data_config = {}
        for i in data_attr:
            data_value = sorted(summary[i].keys())
            data_config[i] = data_value
        return self.remove_outliers(data_config)


    # 列ごとの出現頻度を計算
    def summarize_column_counts(self, array):  # self を追加
        summary = {}
        for col_idx in range(array.shape[1]):
            col_data = array[:, col_idx]
            counts = Counter(col_data)
            summary["{}".format(col_idx)] = dict(counts)
        return summary

    # 外れ値を取り除く関数
    def remove_outliers(self,data):
        filtered_data = {}
        for attr, values in data.items():
            values = np.array(values)
            Q1 = np.percentile(values, 25)  # 25パーセンタイル（第1四分位数）
            Q3 = np.percentile(values, 75)  # 75パーセンタイル（第3四分位数）
            IQR = Q3 - Q1  # 四分位範囲
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # 範囲内の値のみを残す
            filtered_values = values[(values >= lower_bound) & (values <= upper_bound)]
            filtered_data[attr] = filtered_values.tolist()
        return filtered_data