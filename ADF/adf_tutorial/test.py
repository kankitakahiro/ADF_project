# main.py
from my_utils.dataset_config import dataset_config
import numpy as np
import tensorflow as tf
import sys, os

# クラスをインスタンス化
dataset = "census"
config = dataset_config(dataset)

# メソッドを呼び出し
print(config.anlz_dataset())