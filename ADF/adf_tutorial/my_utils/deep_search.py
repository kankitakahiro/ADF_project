
import itertools
import numpy as np
import random
from bisect import bisect_left, bisect_right
import adf_utils.config
from my_utils.clip import clip
from adf_utils.utils_tf import model_prediction, model_argmax
from my_utils.calculate_column_frequencies import calculate_column_frequencies
from itertools import combinations

def generate_binary_vector(indices, length):
    """
    指定されたインデックス位置に 1 を入れ、それ以外の位置に 0 を入れる
    """
    binary_vector = [0] * length  # 初期化（すべて0）
    for index in indices:
        if index < length:  # インデックスが範囲内であることを確認
            binary_vector[index] = 1
    return binary_vector

def reduce_g_diff_and_search(sess, x, preds, g_diff, sample, s_grad, data_config, dataset, perturbation_size, sensitive_param,origin_label):
    """
    Explore all possible patterns of g_diff with 0/1 combinations.
    :param sess: TensorFlow session
    :param x: Input placeholder
    :param preds: Model predictions
    :param g_diff: Gradient difference vector (list)
    :param sample: Original sample
    :param s_grad: Original gradient
    :param data_config: Dataset configuration
    :param dataset: Dataset name
    :param perturbation_size: Size of perturbation
    :param sensitive_param: Index of sensitive parameter
    :param origin_label : 
    :return: Discriminatory sample if found, otherwise None
    """

    ones_indices = [i for i, val in enumerate(g_diff) if val != 0.0]
    deep_serach_iter_count = 0
    ds_iter = 10
    if not ones_indices:
        # If g_diff has no 1s, there's nothing to explore
        return None

    dirs = [list(combinations(ones_indices,r)) for r in range(1,len(ones_indices) + 1)]
    dirs = [list(sublist) for g in dirs for sublist in g]
    while ds_iter > 0 and len(dirs) > 0:
        ds_iter = ds_iter - 1
        comb_label = random.randrange(len(dirs))
        combination = dirs[comb_label]
        dir_comb = generate_binary_vector(combination,len(g_diff))
        # Generate the perturbed sample
        cal_grad = s_grad * np.array(dir_comb)
        # print("cal_grad : ",cal_grad)
        perturbed_sample = sample[0] + perturbation_size * cal_grad
        perturbed_sample = clip(perturbed_sample[0], data_config[dataset])
        perturbed_sample = np.array(perturbed_sample).reshape(1, -1)
        # print('perturbed_sample : ', perturbed_sample)

        # Check model predictions
        probs = model_prediction(sess, x, preds, perturbed_sample)[0]
        label = np.argmax(probs)
        for i in range(data_config[dataset].input_bounds[sensitive_param-1][0], data_config[dataset].input_bounds[sensitive_param-1][1] + 1):
            # 元のサンプルの保護パラメーターと異なる値を試す
            if i != perturbed_sample[0][sensitive_param-1]:
                # サンプルのコピーを作成して保護パラメーターの値を変更
                n_sample = perturbed_sample.copy()
                n_sample[0][sensitive_param-1] = i
                # モデルの予測を取得
                n_probs = model_prediction(sess, x, preds, n_sample)[0]

                # 予測ラベルを取得
                n_label = np.argmax(n_probs)
                # 予測確率を取得
                n_prob = n_probs[n_label]
                
                # もしラベルが変わった場合、n_valueに新しい値を設定しループを終了
                if label != n_label:
                    n_value = i
                    break
        if label != n_label:

            # print("Found discriminatory sample: {}".format(perturbed_sample))
            return perturbed_sample, deep_serach_iter_count

    # print("No discriminatory sample found in all combinations.")
    return None, deep_serach_iter_count

def sublist(lst1, lst2):
    return set(lst1) <= set(lst2)


def reduce_g_diff_and_search_cutoff(sess, x, preds, g_diff, sample, s_grad, data_config, dataset, perturbation_size, sensitive_param,origin_label):
    """
    Explore all possible patterns of g_diff with 0/1 combinations.
    :param sess: TensorFlow session
    :param x: Input placeholder
    :param preds: Model predictions
    :param g_diff: Gradient difference vector (list)
    :param sample: Original sample
    :param s_grad: Original gradient
    :param data_config: Dataset configuration
    :param dataset: Dataset name
    :param perturbation_size: Size of perturbation
    :param sensitive_param: Index of sensitive parameter
    :param origin_label : 
    :return: Discriminatory sample if found, otherwise None
    """

    ones_indices = [i for i, val in enumerate(g_diff) if val != 0.0]
    deep_serach_iter_count = 0
    ds_iter = 10
    if not ones_indices:
        # If g_diff has no 1s, there's nothing to explore
        return None

    dirs = [list(sublist) for r in range(1, len(ones_indices) + 1) for sublist in combinations(ones_indices, r)]
    # print(dirs)

    while ds_iter > 0 and len(dirs) > 0:
        ds_iter = ds_iter - 1
        comb_label = random.randrange(len(dirs))
        selected_dir = dirs[comb_label]
        # Generate binary vector more efficiently
        dir_comb = np.zeros(len(g_diff))
        dir_comb[list(selected_dir)] = 1
        # Generate the perturbed sample
        cal_grad = s_grad * np.array(dir_comb)
        # print("cal_grad : ",cal_grad)
        perturbed_sample = sample[0] + perturbation_size * cal_grad
        perturbed_sample = clip(perturbed_sample[0], data_config[dataset])
        perturbed_sample = np.array(perturbed_sample).reshape(1, -1)
        # print('perturbed_sample : ', perturbed_sample)

        # Check model predictions
        probs = model_prediction(sess, x, preds, perturbed_sample)[0]
        label = np.argmax(probs)
        for i in range(data_config[dataset].input_bounds[sensitive_param-1][0], data_config[dataset].input_bounds[sensitive_param-1][1] + 1):
            # 元のサンプルの保護パラメーターと異なる値を試す
            if i != perturbed_sample[0][sensitive_param-1]:
                # サンプルのコピーを作成して保護パラメーターの値を変更
                n_sample = perturbed_sample.copy()
                n_sample[0][sensitive_param-1] = i
                # モデルの予測を取得
                n_probs = model_prediction(sess, x, preds, n_sample)[0]

                # 予測ラベルを取得
                n_label = np.argmax(n_probs)
                # 予測確率を取得
                n_prob = n_probs[n_label]
                
                # もしラベルが変わった場合、n_valueに新しい値を設定しループを終了
                if label != n_label:
                    n_value = i
                    break
        if label != n_label:
            # print("Found discriminatory sample: {}".format(perturbed_sample))
            return perturbed_sample, deep_serach_iter_count
        to_remove = []
        to_add    = []
        if origin_label == label and origin_label == n_label:
            for i,sub_dir in enumerate(dirs):
                if len(selected_dir) <= len(sub_dir):
                    to_add = to_add + dirs[i:]
                    break
                if not sublist(sub_dir,selected_dir):
                    to_add.append(sub_dir)
                else:
                    to_remove.append(sub_dir)
                    # print(sub_dir)
                    # print(selected_dir)
        else:
            for i in range(len(dirs) - 1, -1, -1):  
                sub_dir = dirs[i]

                if len(selected_dir) >= len(sub_dir):  
                    to_add = dirs[:i + 1] + to_add  
                    break

                if sublist(selected_dir, sub_dir):
                    to_remove.append(sub_dir)
                else:
                    to_add.append(sub_dir)
        dirs = to_add
        # print("to_remove",to_remove)
        # print("to_add",to_add)
    # print("No discriminatory sample found in all combinations.")
    return None, deep_serach_iter_count


# def reduce_g_diff_and_search_cutoff(sess, x, preds, g_diff, sample, s_grad,
#                                       data_config, dataset, perturbation_size,
#                                       sensitive_param, origin_label):
#     """
#     Explore binary patterns of g_diff using bitmasks to reduce computational cost.
    
#     :param sess: TensorFlow session
#     :param x: Input placeholder
#     :param preds: Model predictions
#     :param g_diff: Gradient difference vector (list of floats)
#     :param sample: Original sample (numpy array)
#     :param s_grad: Original gradient (numpy array)
#     :param data_config: Dataset configuration (provides input_bounds, etc.)
#     :param dataset: Dataset name
#     :param perturbation_size: Size of perturbation (float)
#     :param sensitive_param: Index of sensitive parameter (1-indexed)
#     :param origin_label: The original label value used for comparison
#     :return: Tuple (discriminatory_sample, iterations) if found, otherwise (None, iterations)
#     """
#     import numpy as np
#     import random
    
#     # --- Step 1: 前処理 ---
#     # g_diff のうち、0でない要素のインデックスを取得
#     ones_indices = [i for i, val in enumerate(g_diff) if val != 0.0]
#     if not ones_indices:
#         # g_diff に有効な成分がない場合は探索不要
#         return None, 0

#     n = len(ones_indices)  # 対象ビット数
#     # 全候補は 0 から 2^n - 1 の整数（ビットマスク）として生成
#     all_masks = list(range(2 ** n))
#     # 1 の数が少ない順にソート（popcount の小さいものが先頭に来る）
#     sorted_masks = sorted(all_masks, key=lambda mask: bin(mask).count("1"))

#     deep_search_iter_count = 0
#     max_iterations =   # ループ回数の上限（必要に応じて調整）
    
#     # --- Step 2: 探索ループ ---
#     while max_iterations > 0 and sorted_masks:
#         max_iterations -= 1
#         deep_search_iter_count += 1
        
#         # ランダムなインデックスを取得し、その要素をリストから削除して取得する
#         target_mask = sorted_masks.pop(random.randrange(len(sorted_masks)))
        
#         # 新しい g_diff を作成（選択したビットマスクに従い、g_diff の各要素をゼロまたは元の値に）
#         new_g_diff = list(g_diff)
#         # ones_indices に対応する部分だけ、target_mask のビットに合わせて値を残す／ゼロにする
#         for j, index in enumerate(ones_indices):
#             # target_mask の j 番目のビットが 0 なら、その位置はゼロに
#             if not ((target_mask >> j) & 1):
#                 new_g_diff[index] = 0.0

#         # --- Step 3: 変化後のサンプル作成 ---
#         # 勾配と g_diff の積により、perturbed_sample を生成
#         cal_grad = s_grad * np.array(new_g_diff)
#         perturbed_sample = sample[0] + perturbation_size * cal_grad
#         # clip 関数はデータの範囲を data_config[dataset] に基づいて制限するものとする
#         perturbed_sample = clip(perturbed_sample[0], data_config[dataset])
#         perturbed_sample = np.array(perturbed_sample).reshape(1, -1)

#         # --- Step 4: モデル予測と敏感パラメータのチェック ---
#         probs = model_prediction(sess, x, preds, perturbed_sample)[0]
#         label = np.argmax(probs)
#         discriminatory_found = False
#         # sensitive_param は 1-indexed として、変更可能な範囲を data_config から取得
#         lower_bound, upper_bound = data_config[dataset].input_bounds[sensitive_param-1]
#         for i in range(lower_bound, upper_bound + 1):
#             if i == perturbed_sample[0][sensitive_param-1]:
#                 continue  # 現在の値は除外
#             n_sample = perturbed_sample.copy()
#             n_sample[0][sensitive_param-1] = i
#             n_probs = model_prediction(sess, x, preds, n_sample)[0]
#             n_label = np.argmax(n_probs)
#             if label != n_label:
#                 discriminatory_found = True
#                 break  # 異なるラベルが確認できたらループ終了

#         if discriminatory_found:
#             # 差別的なサンプルが見つかった場合、結果を返す
#             return perturbed_sample, deep_search_iter_count

#         # --- Step 5: 残候補の絞り込み ---
#         # 現在の候補 target_mask を元に、残りの sorted_masks から除外する候補を決定する
#         # 定義：
#         #  - candidate が target の「部分集合」である ⇨ (candidate & ~target) == 0
#         #  - candidate が target の「上位集合（または同一）」である ⇨ (target & ~candidate) == 0
#         if origin_label == label and origin_label == n_label:
#             # もし、origin_label と label, n_label がすべて同じならば、
#             # ターゲット（target_mask）の部分集合となる候補を除外します。
#             # ※ candidate が target_mask の部分集合であれば、すべてのビットについて
#             #    candidate の1が target_mask の1に含まれているので (candidate & ~target_mask) は 0 になります。
#             filtered_masks = []
#             for candidate in sorted_masks:
#                 # candidate が部分集合ではない場合のみ残す
#                 if (candidate & ~target_mask) != 0:
#                     filtered_masks.append(candidate)
#             sorted_masks = filtered_masks
#         else:
#             # それ以外の場合、つまり origin_label が label と n_label と一致しない場合、
#             # ターゲット（target_mask）の上位集合（または同一）となる候補を除外します。
#             # ※ candidate が target_mask の上位集合であれば、すべてのビットについて
#             #    target_mask の1が candidate の1に含まれているので (target_mask & ~candidate) は 0 になります。
#             filtered_masks = []
#             for candidate in sorted_masks:
#                 # candidate が上位集合（または同一）ではない場合のみ残す
#                 if (target_mask & ~candidate) != 0:
#                     filtered_masks.append(candidate)
#             sorted_masks = filtered_masks
#     # 探索回数内に差別的サンプルが見つからなかった場合
#     return None, deep_search_iter_count


