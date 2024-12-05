# CUDA 11.5を使用するTensorFlow 1.11.0のイメージをベースにします
FROM tensorflow/tensorflow:1.11.0-gpu-py3

# 必要なライブラリをインストール
RUN pip3 install matplotlib joblib openpyxl
