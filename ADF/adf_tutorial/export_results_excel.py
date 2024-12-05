import os
import pandas as pd
import re
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment


def extract_data_from_files(root_dir, output_excel):
    """
    指定されたディレクトリからすべての .txt ファイルを再帰的に探索し、
    データを抽出して Excel に保存する関数。

    Args:
        root_dir (str): データが含まれるルートディレクトリ。
        output_excel (str): 保存する Excel ファイルのパス。
    """
    data_by_sheet = {}  # シートごとのデータを格納する辞書
    summary_data = {}   # Summary データを格納する辞書

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".txt"):
                file_path = os.path.join(dirpath, file)
                with open(file_path, "r") as f:
                    content = f.read()  # ファイル内容を読み取る

                # データの抽出
                extracted_data = parse_file_content(content)
                extracted_data["file_path"] = file_path  # ファイルのパスを追加

                # ファイルパスからシート名を取得
                sheet_name = extract_sheet_name(file_path)

                # シートごとにデータを分類
                if sheet_name not in data_by_sheet:
                    data_by_sheet[sheet_name] = []
                data_by_sheet[sheet_name].append(extracted_data)

                # Summary 用データの整理
                dataset_key = get_dataset_key(sheet_name)
                print(sheet_name)
                method = get_method(sheet_name)
                if dataset_key not in summary_data:
                    summary_data[dataset_key] = {"adf_origin": {}, "adf_deep_search": {}}
                summary_data[dataset_key][method] = extracted_data

    # データを Excel に保存
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        # 各シートにデータを保存
        for sheet_name, data in data_by_sheet.items():
            df = pd.DataFrame(data)

            # 平均を計算して追加
            averages = df.mean(numeric_only=True).to_dict()
            averages["file_path"] = "Average"
            df = pd.concat([df, pd.DataFrame([averages])], ignore_index=True, sort=False)

            # シート名が長すぎる場合は短縮
            truncated_sheet_name = sanitize_sheet_name(sheet_name)[:31]
            df.to_excel(writer, sheet_name=truncated_sheet_name, index=False)

        # Summary シートを作成
        summary_df = create_summary_table(summary_data)
        summary_df.to_excel(writer, sheet_name="Summary", index=True, startrow=2)

    # Excel ファイルにテーブルのマージを適用
    # add_summary_headers(output_excel, "Summary")
    print("The data has been saved to {}.".format(output_excel))


def parse_file_content(content):
    """
    ファイル内容から必要なデータを抽出する関数。

    Args:
        content (str): ファイルの内容。

    Returns:
        dict: 抽出したデータを含む辞書。
    """
    data = {}

    # 正規表現でデータを抽出
    data["hamming_distance_100"] = extract_value(r"hamming_distance\(100\)\s+([\d.]+)", content)
    data["hamming_distance_500"] = extract_value(r"hamming_distance\(500\)\s+([\d.]+)", content)
    data["hamming_distance_1000"] = extract_value(r"hamming_distance\(1000\)\s+([\d.]+)", content)
    data["total_discriminatory_inputs"] = extract_value(r"Total discriminatory inputs of global search- (\d+)", content)
    data["deep_search_success"] = extract_value(r"deep_search_success\s+:\s+(\d+)", content)
    data["deep_search_failed"] = extract_value(r"deep_search_faild\s+:\s+(\d+)", content)
    data["adf_success"] = extract_value(r"adf_success\s+:\s+(\d+)", content)
    data["adf_failed"] = extract_value(r"adf_faild\s+:\s+(\d+)", content)
    data["both_not_cross"] = extract_value(r"both_not_cross\s+:\s+(\d+)", content)
    data["both_cross"] = extract_value(r"both_cross\s+:\s+(\d+)", content)
    data["execution_time"] = extract_value(r"Execution time:\s+([\d.]+) seconds", content)

    return data


def extract_value(pattern, content):
    """
    正規表現パターンでデータを抽出する関数。

    Args:
        pattern (str): 正規表現パターン。
        content (str): テキスト内容。

    Returns:
        float or None: 抽出された値。該当しない場合は None。
    """
    match = re.search(pattern, content)
    return float(match.group(1)) if match else None


def extract_sheet_name(file_path):
    """
    ファイルパスからシート名を抽出する関数。

    Args:
        file_path (str): ファイルのパス。

    Returns:
        str: シート名（`adf_deep_search/dataset=...` 部分）。
    """
    method_match = re.search(r"(adf_deep_search|adf_origin|adf_fly|adf_deep_fly)", file_path)
    method_name = method_match.group(0) if method_match else "unknown_method"

    dataset_match = re.search(r"dataset=.*?_sensparam=\d+", file_path)
    dataset_name = dataset_match.group(0) if dataset_match else "unknown_dataset"

    return "{}_{}".format(method_name, dataset_name)


def sanitize_sheet_name(sheet_name):
    """
    Excel シート名として使用できない文字を置き換える関数。

    Args:
        sheet_name (str): 元のシート名。

    Returns:
        str: サニタイズされたシート名。
    """
    invalid_chars = r'[\\/?*:[]"<>|]'
    sanitized_name = re.sub(invalid_chars, "_", sheet_name)  # 無効な文字を "_" に置換
    return sanitized_name


def get_dataset_key(sheet_name):
    """
    シート名からデータセットキーを抽出する関数。

    Args:
        sheet_name (str): シート名。

    Returns:
        str: データセットキー（例: `census_age`, `bank_age`）。
    """
    dataset_match = re.search(r"dataset=(.*?)_sensparam=(\d+)", sheet_name)
    if dataset_match:
        dataset = dataset_match.group(1)
        sens_param = dataset_match.group(2)
        return "{}_param_{}".format(dataset, sens_param)
    return "unknown_dataset"


def get_method(sheet_name):
    """
    シート名からメソッド名を抽出する関数。

    Args:
        sheet_name (str): シート名。

    Returns:
        str: メソッド名（`adf_origin` または `adf_deep_search`）。
    """
    if "adf_origin" in sheet_name:
        return "adf_origin"
    elif "adf_deep_search" in sheet_name:
        return "adf_deep_search"
    elif "adf_fly" in sheet_name:
        return "adf_fly"
    elif "adf_deep_fly" in sheet_name:
        return "adf_deep_fly"
    return "unknown_method"


def create_summary_table(summary_data):
    """
    Summary テーブルを作成する関数。

    Args:
        summary_data (dict): 各シートのデータ。

    Returns:
        pd.DataFrame: Summary テーブル。
    """
    rows = []
    for dataset, methods in summary_data.items():
        row = {"Dataset": dataset}
        for method, data in methods.items():
            if data:
                for key, value in data.items():
                    row["{}_{}".format(method, key)] = value
        rows.append(row)

    return pd.DataFrame(rows)


# def add_summary_headers(excel_file, sheet_name):
#     """
#     Summary シートのヘッダーをマージする関数。

#     Args:
#         excel_file (str): Excel ファイルのパス。
#         sheet_name (str): 修正するシート名。
#     """
#     wb = load_workbook(excel_file)
#     ws = wb[sheet_name]

#     # 列の範囲を取得
#     max_col = ws.max_column

#     # マージ範囲を計算してヘッダーを追加
#     adf_origin_start = 2
#     adf_origin_end = (max_col // 2) + 1
#     adf_deepsearch_start = adf_origin_end + 1
#     adf_deepsearch_end = max_col

#     ws.merge_cells(start_row=1, start_column=adf_origin_start, end_row=1, end_column=adf_origin_end)
#     ws.merge_cells(start_row=1, start_column=adf_deepsearch_start, end_row=1, end_column=adf_deepsearch_end)

#     ws.cell(row=1, column=adf_origin_start, value="adf_origin").alignment = Alignment(horizontal="center")
#     ws.cell(row=1, column=adf_deepsearch_start, value="adf_deepsearch").alignment = Alignment(horizontal="center")

#     wb.save(excel_file)


if __name__ == "__main__":
    # データのあるフォルダ
    data_folder = "data"
    
    # 保存先の Excel ファイル名
    output_excel_file = "output_data.xlsx"

    # 実行
    extract_data_from_files(data_folder, output_excel_file)
