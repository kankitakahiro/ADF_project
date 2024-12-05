import adf_utils.config

# 差別データの出現頻度を更新する
def calculate_column_frequencies(column_frequencies, disc_data, dataset):
    # config_nameからコンフィグレーションクラスを取得する
    config_name = dataset
    config_class = getattr(adf_utils.config, config_name)
    # コンフィグレーションクラスのインスタンスを作成する
    config_instance = config_class()

    if (column_frequencies == None): 
        column_frequencies = {}
        for name, bounds in zip(config_instance.feature_name, config_instance.input_bounds):
            column_frequencies[name] = {i: 0 for i in range(bounds[0], bounds[1] + 1)}
    
    col_index = 0
    
    for feature_data in disc_data:
        feature_name = config_instance.feature_name[col_index]
        column_frequencies[feature_name][feature_data] += 1
        col_index += 1
    return column_frequencies


