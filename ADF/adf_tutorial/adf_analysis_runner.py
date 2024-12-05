import os
import subprocess


# 対応表：データセット名と `sens_param` の説明
SENS_PARAM_MAPPING = {
    "census": {
        1: "age",
        8: "race",
        9: "sex",
    },
    "credit": {
        9: "sex",
        13: "age",
    },
    "bank": {
        1: "age",
    }
}


def process_adf_multiple_runs(file_paths, output_dir, num_runs, args_list):
    """
    複数のADFファイルを複数回実行し、引数を渡して結果を指定のディレクトリ構造で保存する関数。

    Args:
        file_paths (list of str): 実行するPythonファイルのパスリスト
        output_dir (str): 実行結果を保存するルートディレクトリ
        num_runs (int): 各引数セットの実行回数
        args_list (list of dict): 渡す引数のリスト
    """
    for file_path in file_paths:
        base_name = os.path.basename(file_path).replace(".py", "")
        print("Processing {} for {} argument sets, each {} times...".format(file_path, len(args_list), num_runs))
        
        for script_args in args_list:
            # 出力に使用する引数の値を取得
            dataset = script_args.get("dataset", "unknown")
            sens_param_index = script_args.get("sens_param", "unknown")
            max_global = script_args.get("max_global", "unknown")

            # sens_param の名前を対応表から取得
            sens_param_name = SENS_PARAM_MAPPING.get(dataset, {}).get(sens_param_index, "unknown_param_{}".format(sens_param_index))

            # ディレクトリ名の識別子作成
            dir_identifier = "dataset={}_sensparam={}_{}_maxglobal={}".format(dataset, sens_param_index, sens_param_name, max_global)

            # 実行結果保存ディレクトリを一度に作成
            for run_count in range(1, num_runs + 1):
                run_dir = os.path.join(output_dir, base_name, dir_identifier, "run_{:02}".format(run_count))
                os.makedirs(run_dir, exist_ok=True)

            # 複数回実行
            for run_count in range(1, num_runs + 1):
                run_dir = os.path.join(output_dir, base_name, dir_identifier, "run_{:02}".format(run_count))
                # ファイル名を実行ファイル名、引数、実行回数を含む形式に変更
                result_file = os.path.join(run_dir, "{}_{}_run_{:02}_output.txt".format(base_name, dir_identifier, run_count))

                # コマンドを構築
                command = ["python3", file_path]
                for arg_name, arg_value in script_args.items():
                    command.append("--{}".format(arg_name))
                    command.append("{}".format(arg_value))

                # subprocessを使用してスクリプトを実行し、結果をファイルに保存
                with open(result_file, "w") as output:
                    subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)

                print("Run {} completed for {}, results saved to {}".format(run_count, file_path, result_file))


if __name__ == "__main__":
    # 実行ファイルと設定
    # files_to_process = ["adf_origin.py", "adf_deep_search.py" ,"adf_fly.py","adf_deep_fly.py"]  # 実行するPythonファイルのパスリスト
    files_to_process = ["adf_origin.py", "adf_deep_search.py"]  # 実行するPythonファイルのパスリスト
    output_directory = "data"                                   # 結果を保存するディレクトリ
    num_runs = 5                                                # 各引数セットの実行回数

    # デフォルトの共通引数
    default_arguments = {
        "model_path": "../models/",
        "cluster_num": 4,
        "max_global": 1000,
        "max_local": 100,
        "max_iter": 10,
    }

    # データセットごとの設定
    dataset_arguments = {
        "census": [
            {"sens_param": 1},  # age
            {"sens_param": 8},  # race
            {"sens_param": 9},  # gender
        ],
        "credit": [
            {"sens_param": 9},  # gender
            {"sens_param": 13},  # age
        ],
        "bank": [
            {"sens_param": 1},  # age
        ],
    }

    # すべての引数セットを生成
    arguments = []
    for dataset, specific_args in dataset_arguments.items():
        for args in specific_args:
            # デフォルト引数に特定の設定を上書き
            argument = {**default_arguments, "dataset": dataset, **args}
            arguments.append(argument)

    # 確認用に出力
    for arg in arguments:
        print(arg)

    # 実行
    process_adf_multiple_runs(files_to_process, output_directory, num_runs, arguments)
