
import itertools
import numpy as np
import adf_utils.config
from my_utils.clip import clip
from adf_utils.utils_tf import model_prediction, model_argmax
from my_utils.calculate_column_frequencies import calculate_column_frequencies
def reduce_g_diff_and_search(sess, x, preds, g_diff, sample, s_grad, data_config, dataset, perturbation_size, sensitive_param):
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
    :return: Discriminatory sample if found, otherwise None
    """

    ones_indices = [i for i, val in enumerate(g_diff) if val != 0.0]

    if not ones_indices:
        # If g_diff has no 1s, there's nothing to explore
        return None

    # Generate all possible binary combinations for the ones_indices
    num_combinations = 2 ** len(ones_indices)
    # print("Exploring {} combinations.".format(num_combinations))
    # print(ones_indices)
    combinations = list(itertools.product([0, 1], repeat=len(ones_indices)))

    # 1の数が多い順で並べ替える
    sorted_combinations = sorted(combinations, key=lambda combination: combination.count(1), reverse=True)

    for combination in sorted_combinations:
        # print("conbination : ",combination)
        new_g_diff = g_diff.copy()

        # Apply the combination to the new_g_diff
        for index, bit in zip(ones_indices, combination):
            new_g_diff[index] = bit * g_diff[index]

        # Generate the perturbed sample
        cal_grad = s_grad * np.array(new_g_diff)
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
            return perturbed_sample
    # print("No discriminatory sample found in all combinations.")
    return None


def reduce_g_diff_and_fly(sess, x, preds, g_diff, sample, s_grad, data_config, dataset, perturbation_size, sensitive_param, column_frequencies):
    config_name = dataset
    config_instance = getattr(adf_utils.config, config_name)

    ones_indices = [i for i, val in enumerate(g_diff) if val != 0.0]

    if not ones_indices:
        return None

    combinations = list(itertools.product([0, 1], repeat=len(ones_indices)))
    sorted_combinations = sorted(combinations, key=lambda combination: combination.count(1), reverse=True)

    for combination in sorted_combinations:
        new_g_diff = g_diff.copy()

        for index, bit in zip(ones_indices, combination):
            new_g_diff[index] = bit * g_diff[index]

        cal_grad = s_grad * np.array(new_g_diff)
        perturbed_sample = sample[0] + perturbation_size * cal_grad[0]
        perturbed_sample = clip(perturbed_sample, data_config[dataset])

        diversity_sample = []  # Ensure diversity_sample is a flat list

        for col_index, bounds in enumerate(config_instance.input_bounds):
            feature_name = config_instance.feature_name[col_index]
            if column_frequencies:
                col_freq = column_frequencies[feature_name]
                col_diff = int(cal_grad[0][col_index])
                if col_diff < 0:
                    for value in range(int(perturbed_sample[col_index]), int(bounds[0] - 1), col_diff):
                        if col_freq.get(value, 0) <= col_freq.get(int(perturbed_sample[col_index]), 0):
                            diversity_sample.append(value)
                            break
                elif col_diff > 0:
                    for value in range(int(perturbed_sample[col_index]), int(bounds[1] + 1), col_diff):
                        if col_freq.get(value, 0) <= col_freq.get(int(perturbed_sample[col_index]), 0):
                            diversity_sample.append(value)
                            break
                else:
                    diversity_sample.append(int(perturbed_sample[col_index]))
                if len(diversity_sample) <= col_index:
                    diversity_sample.append(int(perturbed_sample[col_index]))
            else:
                diversity_sample = perturbed_sample.tolist()  # Convert to flat list if column_frequencies is None

        # Convert diversity_sample to a NumPy array, ensuring it's 1D
        diversity_sample = np.array(diversity_sample, dtype=np.float32).flatten()
        # print('fly data     ', diversity_sample)

        # Ensure perturbed_sample is updated correctly
        perturbed_sample = clip(diversity_sample, data_config[dataset])
        perturbed_sample = np.array(perturbed_sample).reshape(1, -1)

        probs = model_prediction(sess, x, preds, perturbed_sample)[0]
        label = np.argmax(probs)
        for i in range(data_config[dataset].input_bounds[sensitive_param-1][0], data_config[dataset].input_bounds[sensitive_param-1][1] + 1):
            if i != perturbed_sample[0][sensitive_param-1]:
                n_sample = perturbed_sample.copy()
                n_sample[0][sensitive_param-1] = i
                n_probs = model_prediction(sess, x, preds, n_sample)[0]
                n_label = np.argmax(n_probs)
                if label != n_label:
                    return perturbed_sample

    return None