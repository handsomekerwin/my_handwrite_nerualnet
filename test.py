import numpy as np


def normalize(array):
    max_number = np.max(np.absolute(array), axis=1, keepdims=True)
    scale_rate = np.where(max_number == 0, 1, 1/max_number)
    norm = array*scale_rate
    return norm

def v_normalize(array):
    max_number = np.max(np.absolute(array))
    scale_rate = np.where(max_number == 0, 1, 1/max_number)
    norm = array*scale_rate
    return norm

def create_weight(n_inputs, n_nerons):
    """
    :param n_inputs: 输入层神经元个数
    :param n_nerous: 输出层神经元个数
    :return: 一个矩阵，矩阵的大小为n_inputs*n_nerons
    """
    weights = np.random.randn(n_inputs, n_nerons) / np.sqrt(n_inputs)
    return weights

def get_weight_adjust_matrix( pre_weight_values, after_weight_demand, ways= 1):
    if ways  == 1 :
        plain_weight_values = pre_weight_values
        plain_weight_values = plain_weight_values[:, np.newaxis]
        plain_weight_values = plain_weight_values.repeat(2, axis=1)
        weight_adjust_matrix= plain_weight_values*after_weight_demand
    else :
        plain_weight_values = np.full((len(pre_weight_values), len(after_weight_demand)), 1.)
        pre_weight_values_T = plain_weight_values.T
        weight_adjust_matrix =  (pre_weight_values_T*pre_weight_values).T * after_weight_demand
    print(weight_adjust_matrix)



if __name__ =="__main__":
    a = np.array([[1., 2, 3, 4],[1,3,5,7]])
    b = np.array([1, 1])
    c = normalize(a)
    print(c)
    pass