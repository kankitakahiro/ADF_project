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

# import random

# def clip(input, conf):
#     """
#     Clip the generating instance with each feature to make sure it is valid
#     :param input: generating instance
#     :param conf: the configuration of dataset
#     :return: a valid generating instance
#     """
#     for i in range(len(input)):
#         lower_bound = conf.input_bounds[i][0]
#         upper_bound = conf.input_bounds[i][1]
#         if input[i] < lower_bound or input[i] > upper_bound:
#             input[i] = random.uniform(lower_bound, upper_bound)
#         else:
#             input[i] = max(input[i], lower_bound)
#             input[i] = min(input[i], upper_bound)
#     return input