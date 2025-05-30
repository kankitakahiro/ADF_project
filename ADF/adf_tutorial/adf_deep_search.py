import numpy as np
import tensorflow as tf
import sys, os
sys.path.append("../")
import copy
import time
import itertools

from tensorflow.python.platform import flags
from scipy.optimize import basinhopping

from adf_data.census import census_data
from adf_data.credit import credit_data
from adf_data.bank import bank_data
from adf_model.tutorial_models import dnn
from adf_utils.utils_tf import model_prediction, model_argmax
from adf_utils.config import census, credit, bank
from adf_tutorial.utils import cluster, gradient_graph

from my_utils.calculate_hamming_distances import hamming_distance_sum
from my_utils.deep_search import reduce_g_diff_and_search
from my_utils.deep_search import reduce_g_diff_and_search_cutoff
from my_utils.dataset_config import dataset_config

FLAGS = flags.FLAGS


def define_perturbation(dataset, sens_param):
    k = 10
    dataset_anlz = dataset_config(dataset)
    dataset_anlz_data = dataset_anlz.anlz_dataset()
    # キーを数値に変換してソート
    sorted_keys = sorted(dataset_anlz_data.keys(), key=int)
    # ソート順に従って、各キーのリストから最大値を取得
    max_values = [max(dataset_anlz_data[k]) for k in sorted_keys]
    perturbation_size = [max(1, round(val / k)) for val in max_values]
    if dataset == "census":
        # インデックス 2, 9, 10, 11 以外をすべて 1 に上書き
        for i in range(len(perturbation_size)):
            if i not in [2, 9, 10, 11]:
                perturbation_size[i] = 1
    elif dataset == "bank":
        # インデックス 5,9,11,12,13 以外をすべて 1 に上書き
        for i in range(len(perturbation_size)):
            if i not in [5,9,11,12,13]:
                perturbation_size[i] = 1
    elif dataset == "credit":
        # インデックス 1,4 以外をすべて 1 に上書き
        for i in range(len(perturbation_size)):
            if i not in [1,4]:
                perturbation_size[i] = 1
    # print("key:", sorted_keys)
    # print("max list:", max_values)
    # print("sensitive_param:", sens_param)

    # print(dataset_anlz_data)
    # print(perturbation_size)
    return perturbation_size

def check_for_error_condition(conf, sess, x, preds, t, sens):
    """
    Check whether the test case is an individual discriminatory instance
    :param conf: the configuration of dataset
    :param sess: TF session
    :param x: input placeholder
    :param preds: the model's symbolic output
    :param t: test case
    :param sens: the index of sensitive feature
    :return: whether it is an individual discriminatory instance
    """
    t = t.astype('int')
    label = model_argmax(sess, x, preds, np.array([t]))

    # check for all the possible values of sensitive feature
    for val in range(conf.input_bounds[sens-1][0], conf.input_bounds[sens-1][1]+1):
        if val != t[sens-1]:
            tnew = copy.deepcopy(t)
            tnew[sens-1] = val
            label_new = model_argmax(sess, x, preds, np.array([tnew]))
            if label_new != label:
                return True
    return False

def seed_test_input(clusters, limit):
    """
    Select the seed inputs for fairness testing
    :param clusters: the results of K-means clustering
    :param limit: the size of seed inputs wanted
    :return: a sequence of seed inputs
    """
    i = 0
    rows = []
    max_size = max([len(c[0]) for c in clusters])
    while i < max_size:
        if len(rows) == limit:
            break
        for c in clusters:
            if i >= len(c[0]):
                continue
            row = c[0][i]
            rows.append(row)
            if len(rows) == limit:
                break
        i += 1
    return np.array(rows)

def clip(input, conf):
    """
    Clip the generating instance with each feature to make sure it is valid
    :param input: generating instance
    :param conf: the configuration of dataset
    :return: a valid generating instance
    """
    for i in range(len(input)):
        input[i] = max(input[i], conf.input_bounds[i][0])
        input[i] = min(input[i], conf.input_bounds[i][1])
    return input

def dnn_fair_testing(dataset, sensitive_param, model_path, cluster_num, max_global, max_local, max_iter):
    """
    The implementation of ADF
    :param dataset: the name of testing dataset
    :param sensitive_param: the index of sensitive feature
    :param model_path: the path of testing model
    :param cluster_num: the number of clusters to form as well as the number of
            centroids to generate
    :param max_global: the maximum number of samples for global search
    :param max_local: the maximum number of samples for local search
    :param max_iter: the maximum iteration of global perturbation
    """
    data = {"census":census_data, "credit":credit_data, "bank":bank_data}
    data_config = {"census":census, "credit":credit, "bank":bank}

    # step size of perturbation
    perturbation_size = define_perturbation(dataset,sensitive_param)

    # prepare the testing data and model
    X, Y, input_shape, nb_classes = data[dataset]()
    tf.set_random_seed(1234)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=config)
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    model = dnn(input_shape, nb_classes)
    preds = model(x)
    saver = tf.train.Saver()
    model_path = model_path + dataset + "/test.model"
    saver.restore(sess, model_path)

    # construct the gradient graph
    grad_0 = gradient_graph(x, preds)

    # build the clustering model
    clf = cluster(dataset, cluster_num)
    clusters = [np.where(clf.labels_==i) for i in range(cluster_num)]

    # store the result of fairness testing
    # tot_inputs = set()
    global_disc_inputs = set()
    global_disc_inputs_list = []
    value_list = []
    suc_idx = []
    adf_iter_count = 0
    both_not_cross = 0
    both_cross     = 0
    deep_search_success = 0
    deep_search_faild   = 0
    adf_success         = 0
    adf_faild           = 0
    deepsearch_time     = []
    deep_search_iter_count = []

    # select the seed input for fairness testing
    inputs = seed_test_input(clusters, min(max_global, len(X)))
    # config_nameからコンフィグレーションクラスを取得
    config_class = data_config[dataset]
    # コンフィグレーションクラスのインスタンスを作成
    config_instance = config_class()

    for num in range(len(inputs)):
        index = inputs[num]
        sample = X[index:index+1]

        # 最初のラベルを計算する
        origin_probs = model_prediction(sess, x, preds, sample)[0]
        origin_label = np.argmax(origin_probs)
        previous_sample = sample
        deep_flag = False

        # start global perturbation
        for iter in range(max_iter+1):
            probs = model_prediction(sess, x, preds, sample)[0]
            label = np.argmax(probs)
            prob = probs[label]
            max_diff = 0
            n_value = -1
            adf_iter_count += 1

            # search the instance with maximum probability difference for global perturbation
            for i in range(config_instance.input_bounds[sensitive_param-1][0], config_instance.input_bounds[sensitive_param-1][1] + 1):
                if i != sample[0][sensitive_param-1]:
                    n_sample = sample.copy()
                    n_sample[0][sensitive_param-1] = i
                    n_probs = model_prediction(sess, x, preds, n_sample)[0]
                    n_label = np.argmax(n_probs)
                    n_prob = n_probs[n_label]
                    if label != n_label:
                        n_value = i
                        break
                    else:
                        prob_diff = abs(prob - n_prob)
                        if prob_diff > max_diff:
                            max_diff = prob_diff
                            n_value = i

            temp = copy.deepcopy(sample[0].astype('int').tolist())
            temp = temp[:sensitive_param - 1] + temp[sensitive_param:]

            # if get an individual discriminatory instance
            if label != n_label and (tuple(temp) not in global_disc_inputs):
                if (deep_flag):
                    deep_search_success += 1
                else:
                    adf_success += 1
                global_disc_inputs_list.append(temp)
                global_disc_inputs.add(tuple(temp))
                value_list.append([sample[0, sensitive_param - 1], n_value])
                suc_idx.append(index)
                break

            n_sample[0][sensitive_param - 1] = n_value

            if iter == max_iter:
                if origin_label == label and origin_label == n_label:
                    # print('Both do not cross boundaries.')
                    both_not_cross += 1
                else:
                    # print('Both crossed boundaries.')
                    both_cross     += 1
                adf_faild += 1
                break

            # global perturbation
            s_grad = sess.run(tf.sign(grad_0), feed_dict={x: sample})
            n_grad = sess.run(tf.sign(grad_0), feed_dict={x: n_sample})

            # find the feature with same impact
            if np.zeros(data_config[dataset].params).tolist() == s_grad[0].tolist():
                g_diff = n_grad[0]
            elif np.zeros(data_config[dataset].params).tolist() == n_grad[0].tolist():
                g_diff = s_grad[0]
            else:
                g_diff = np.array(s_grad[0] == n_grad[0], dtype=float)
            g_diff[sensitive_param - 1] = 0
            if np.zeros(input_shape[1]).tolist() == g_diff.tolist():
                index = np.random.randint(len(g_diff) - 1)
                if index > sensitive_param - 2:
                    index = index + 1
                g_diff[index] = 1.0

            if origin_label == label and origin_label == n_label:
                # print('Both do not cross boundaries.')
                if np.zeros(input_shape[1]).tolist() == g_diff.tolist():
                    index = np.random.randint(len(g_diff) - 1)
                    if index > sensitive_param - 2:
                        index = index + 1
                    g_diff[index] = 1.0

                cal_grad = s_grad * g_diff
                # print(cal_grad)
                # print(perturbation_size)
                # print(sample[0])
                # 更新前のデータを保存する。両方決定境界を超えたときに使用する
                previous_sample = sample
                sample[0] = clip(sample[0] + perturbation_size * cal_grad[0], data_config[dataset]).astype("int")
            else:
                # print('Both crossed boundaries.')
                # Use recursive reduction of g_diff
                deepsearch_start_time = time.time()
                deep_flag = True
                # HACK 関数名でcutoffのあるなしを変更している。
                # reduce_g_diff_and_search        cutoffなし
                # reduce_g_diff_and_search_cutoff cutoffあり
                result, each_deep_search_iter_count = reduce_g_diff_and_search_cutoff(sess, x, preds, g_diff, previous_sample, s_grad, data_config, dataset, perturbation_size, sensitive_param,origin_label)
                deep_search_iter_count.append(each_deep_search_iter_count)
                deepsearch_end_time = time.time()
                deepsearch_time.append(deepsearch_end_time - deepsearch_start_time)
                if result is not None:
                    sample = result
                else:
                    # print("No discriminatory data found after recursive search.")
                    deep_search_faild += 1
                    both_cross     += 1
                    break
    # create the folder for storing the fairness testing result
    if not os.path.exists('../results/'):
        os.makedirs('../results/')
    if not os.path.exists('../results/' + dataset + '/'):
        os.makedirs('../results/' + dataset + '/')
    if not os.path.exists('../results/'+ dataset + '/'+ str(sensitive_param) + '/'):
        os.makedirs('../results/' + dataset + '/'+ str(sensitive_param) + '/')

    # storing the fairness testing result
    np.save('../results/'+dataset+'/'+ str(sensitive_param) + '/suc_idx.npy', np.array(suc_idx))
    np.save('../results/'+dataset+'/'+ str(sensitive_param) + '/global_samples.npy', np.array(global_disc_inputs_list))
    np.save('../results/'+dataset+'/'+ str(sensitive_param) + '/disc_value.npy', np.array(value_list))

    # print the overview information of result
    # print("Total Inputs are " + str(len(tot_inputs)))
    hamming_distance = hamming_distance_sum(global_disc_inputs_list,100)
    print('hamming_distance(100)',hamming_distance)
    hamming_distance = hamming_distance_sum(global_disc_inputs_list,300)
    print('hamming_distance(300)',hamming_distance)
    hamming_distance = hamming_distance_sum(global_disc_inputs_list,500)
    print('hamming_distance(500)',hamming_distance)
    hamming_distance = hamming_distance_sum(global_disc_inputs_list)
    print('hamming_distance(1000)',hamming_distance)
    hamming_distance = hamming_distance_sum(global_disc_inputs_list,1500)
    print('hamming_distance(1500)',hamming_distance)
    num_global_disc_inputs = len(global_disc_inputs)
    print("Total discriminatory inputs of global search- " + str(num_global_disc_inputs))
    # FIXME:both_crossの数がADFとDeep Searchの差ができる部分を改良する
    print('both_not_cross : ', both_not_cross)
    print('both_cross     : ', both_cross    )
    num_total_attempts = 600 if dataset == "credit" and max_global >= 600 else max_global
    # assert both_cross + both_not_cross + num_global_disc_inputs == num_total_attempts
    print('deep_search_success : ', deep_search_success)
    print('deep_search_faild   : ', deep_search_faild  )
    print('adf_success         : ', adf_success        )
    print('adf_faild           : ', adf_faild          )
    print('check result sum    : ', both_cross + both_not_cross + num_global_disc_inputs,num_total_attempts)
    print('adf_iter_count      : ', adf_iter_count)
    print('deep_search_time    : ', sum(deepsearch_time))
    print('deep_search_iter_count : ',sum(deep_search_iter_count))
    # print('deep_search_iter_count : ', deep_search_iter_count)

def main(argv=None):
    start_time = time.time()
    dnn_fair_testing(dataset = FLAGS.dataset,
                     sensitive_param = FLAGS.sens_param,
                     model_path = FLAGS.model_path,
                     cluster_num=FLAGS.cluster_num,
                     max_global=FLAGS.max_global,
                     max_local=FLAGS.max_local,
                     max_iter = FLAGS.max_iter)
    end_time = time.time()
    print("Execution time: {} seconds".format(end_time - start_time))



if __name__ == '__main__':
    flags.DEFINE_string("dataset", "credit", "the name of dataset")
    flags.DEFINE_integer('sens_param', 9, 'sensitive index, index start from 1, 9 for gender, 8 for race')
    flags.DEFINE_string('model_path', '../models/', 'the path for testing model')
    flags.DEFINE_integer('cluster_num', 4, 'the number of clusters to form as well as the number of centroids to generate')
    flags.DEFINE_integer('max_global', 100, 'maximum number of samples for global search')
    flags.DEFINE_integer('max_local', 100, 'maximum number of samples for local search')
    flags.DEFINE_integer('max_iter', 10, 'maximum iteration of global perturbation')

    tf.app.run()
