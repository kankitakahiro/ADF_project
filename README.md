# ADF_project

自分のgithubで管理することを推奨

先行研究 : https://github.com/pxzhang94/ADF

このコード : https://github.com/kankitakahiro/ADF_project

### コンテナの起動

```bash
# コンテナの起動
sudo docker run -it --gpus all -v ./ADF:/adf adf-master bash
# コンテナに入る
sudo docker container exec -it 4f666b029d43 bash
```

```/adf/adf_tutorial```直下にコードが置いてある。

```/myutil```下には自作した頻繁に使う関数を置いてある。

```adf_analysis_runner.py``` 一括でデータを取るためのプログラム

```export_results_excel.py``` 取得したデータを分析するプログラム

```adf_deep_search.py``` 提案手法の実装されたプログラム。決定境界付近を詳しく探索する。cutoff機能のあるなしを関数名の書き換えで行っている。

```adf_fly.py``` 現在使っていない。提案手法の実装されたプログラム。多様性を向上させるために既知の差別データを参照しながら探索を行う。

```adf_deep_fly``` 現在使っていない。提案手法の実装されたプログラム。上２つの提案手法を両方実装した物。

```adf_origin.py``` 先行研究のプログラム。必要なデータを取得できるように変更したもの

```dnn_tutorial.py``` 本当になにも変更していないプログラム

### 課題

500個のデータを探索すると500個の差別データが発見されるなどのおかしな動作をする。

解決方法 : 一度先行研究をクローンし直して少しずつ実装を行い動作を認認する。

時間がかかる

解決方法 : 無駄なアルゴリズムがあると考えられるため計算量を考えて変更を加える。