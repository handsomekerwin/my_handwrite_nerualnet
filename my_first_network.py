import copy
import math

import numpy as np
from colorama import Fore, Back, Style

import createDataAndPlot as cdp

BATCHSIZE = 50
NETWORK_SHAPE = [2, 4, 5, 2]
LEARNING_RATE = 0.03 # 0.003

def normalize(array):
    max_number = np.max(np.absolute(array), axis=1, keepdims=True)
    scale_rate = 1/np.where(max_number == 0, 1, max_number)
    norm = array*scale_rate
    return norm

def v_normalize(array):
    max_number = np.max(np.absolute(array))
    scale_rate = 1/np.where(max_number == 0, 1, max_number)
    norm = array*scale_rate
    return norm

# def normalize(array):
#     max_number = np.max(np.absolute(array),axis=1,keepdims=True)
#     scale_rate = np.where(max_number==0,1,1/max_number)
#     norm = array*scale_rate
#     return norm

def activation_Relu(x):
 return np.maximum(x, 0)


def get_last_layer_preAct_demands(pred, label):
    """
    :param pred: 预测值
    :param label: 标签
    :return: 一个矩阵，矩阵的大小为len(label)*2
    """
    demand_matrix = np.zeros((len(label), 2))
    demand_matrix[:, 1] = label
    demand_matrix[:, 0] = 1-label
    for i in range(len(label)):
        s = pred[i] * demand_matrix[i]
        j = np.sum(s, axis=0)
        if j > 0.5:
            demand_matrix[i,:] = 0
        else:
            demand_matrix[i] = (demand_matrix[i]-0.5)*2
    return demand_matrix


def precise_loss(pred, label):
    label_matrix = np.zeros((len(label), 2))
    label_matrix[:, 1] = label
    label_matrix[:, 0] = 1 - label
    product = np.sum(pred * label_matrix, axis=1)
    return 1 - product

def loss_function(pred, label):
    condition = (pred > 0.5)
    binary_pred = np.where(condition, 1, 0)
    label_matrix = np.zeros((len(label), 2))
    label_matrix[:, 1] = label
    label_matrix[:, 0] = 1 - label
    product = np.sum(binary_pred * label_matrix, axis=1)
    return 1 - product

def create_weight(n_inputs, n_nerons):
    """
    :param n_inputs: 输入层神经元个数
    :param n_nerous: 输出层神经元个数
    :return: 一个矩阵，矩阵的大小为n_inputs*n_nerons
    """
    weights = np.random.randn(n_inputs, n_nerons) / np.sqrt(n_inputs)
    return weights

def create_bias(n_nerons):
    return np.random.randn(n_nerons)

def classify(x):
    return np.rint(x[:, 1])


class Layer:
    def __init__(self, n_inputs, n_nerons):
        self.weights = create_weight(n_inputs, n_nerons)
        self.bias = create_bias(n_nerons)
        self.activation = activation_Relu


    def forward(self, inputs):
        sum1 = np.dot(inputs, self.weights) + self.bias
        # self.output = self.activation(sum1)
        return sum1


    def layer_backward(self, preweight_values, afterweight_demands):
        pre_weight_demand = np.dot(afterweight_demands, self.weights.T)
        condition = (preweight_values > 0)
        value_derivative = np.where(condition, 1, 0) # relu
        preActs_demand = value_derivative * pre_weight_demand  # mask
        norm_preActs_demand = normalize(preActs_demand)
        weight_adjust_matrix = self.get_weight_adjust_matrix(preweight_values, afterweight_demands)
        norm_weight_adjust_matrix = normalize(weight_adjust_matrix)
        return norm_preActs_demand, norm_weight_adjust_matrix
        pass

    def get_weight_adjust_matrix(self, pre_weight_values, after_weight_demand):
        plain_weight_values = np.full((self.weights.shape), 1.)
        pre_weight_values_T = plain_weight_values.T
        weight_adjust_matrix = np.full(self.weights.shape, 0.)
        for i in range(BATCHSIZE):
            weight_adjust_matrix += (pre_weight_values_T * pre_weight_values[i, :]).T * after_weight_demand[i, :]
        weight_adjust_matrix = weight_adjust_matrix/BATCHSIZE
        return weight_adjust_matrix


def activation_softmax(input):
    max = np.max(input, axis=1, keepdims=True)
    slided_input = input-max
    exp_input = np.exp(slided_input)
    sum_input = np.sum(exp_input, axis=1, keepdims=True)
    output = exp_input/sum_input
    return output



class Network:
    def __init__(self, network_shape):
        self.layers = []
        for i in range(len(network_shape)-1):
            self.layers.append(Layer(network_shape[i], network_shape[i+1]))
        self.shape = network_shape
        self.relu =  activation_Relu
        self.softmax = activation_softmax
        self.stop_train = False

    def precise_loss(self, pred, label):
        label_matrix = np.zeros((len(label), 2))
        label_matrix[:, 1] = label
        label_matrix[:, 0] = 1 - label
        product = np.sum(pred * label_matrix, axis=1)
        return 1 - product

    def forward(self, x):
        output = [x]
        for layer,i in zip(self.layers, np.arange(len(self.layers))):
            if i < len(self.layers)-1: 
                sum_output_ = layer.forward(output[-1])
                new_output = self.relu(sum_output_)

            else:
                sum_output_ = layer.forward(output[-1])
                new_output = self.softmax(sum_output_)

            output.append(new_output)
        return output

    def network_backward(self, layer_outputs, target_vecter):
        backup_network = copy.deepcopy(self)
        preAct_demands = get_last_layer_preAct_demands(layer_outputs[-1], target_vecter)
        for i in range(len(self.layers)):
            layer = backup_network.layers[len(self.layers) - (1+i)]
            if i != 0:
                layer.bias += LEARNING_RATE * np.mean(preAct_demands,axis=0)
                layer.bias = v_normalize(layer.bias)
            output_ = layer_outputs[len(self.layers) - (1+i)]
            norm_preActs_demand, norm_weight_adjust_matrix = layer.layer_backward(output_, preAct_demands)
            preAct_demands = norm_preActs_demand
            layer.weights += norm_weight_adjust_matrix*LEARNING_RATE
            layer.weights = normalize(layer.weights)
        return backup_network

    def train(self, entry=100000):
        global force_update, random_update, n_improved, n_not_improved
        force_update = False
        random_update = False
        n_improved = 0.000000000000001
        n_not_improved = 0.00000000000000001
        epoch = math.floor(entry/BATCHSIZE)
        for i in range(epoch):
            train_data = cdp.create_data(BATCHSIZE)
            self.one_batch_train(train_data)
            improve_rate = n_improved/(n_not_improved+n_improved)
            print(Fore.CYAN+f'improve_rate: {improve_rate:.4f}%')
            if improve_rate <= 0.1:
                force_update = True
            if improve_rate <= 0.01:
                random_update = True
            if self.stop_train == True:
                break
        if self.stop_train ==False:
            print(Fore.RED + " Congratulation, Complete training without true_loss" + Fore.RESET)

    def one_batch_train(self, batch):
        global  force_update, n_improved, n_not_improved, random_update
        input = batch[:, (0, 1)]
        label = copy.deepcopy(batch[:, 2]).astype(int)
        output = self.forward(input)
        precise_loss = loss_function(output[-1], label)
        true_loss = self.precise_loss(output[-1], label)
        if np.mean(true_loss) <= 0.10:
            self.stop_train = True
            print(Fore.RED+f" Congratulation, Complete training with true loss: {np.mean(true_loss)}"+Fore.RESET)
        else:
            backup_network = self.network_backward(output, label)
            backup_output = backup_network.forward(input)
            backup_precise_loss = loss_function(backup_output[-1], label)
            if np.mean(precise_loss) >= np.mean(backup_precise_loss):
                for i in range(len(self.layers)):
                    self.layers[i].weights = backup_network.layers[i].weights.copy()
                    self.layers[i].bias = backup_network.layers[i].bias.copy()
                # print(Fore.GREEN+" Improved" + Fore.RESET)
                n_improved += 1
            elif force_update == True:
                for i in range(len(self.layers)):
                    self.layers[i].weights = backup_network.layers[i].weights.copy()
                    self.layers[i].bias = backup_network.layers[i].bias.copy()
                print(Fore.GREEN + "-----------------Force Improved------------" + Fore.RESET)
                n_improved = 0.000000000000001
                n_not_improved = 0.00000000000000001
                force_update = False
            elif random_update == True:
                self.random_update()
                print(Fore.RED + "---------------------------Random Improved--------------------------------" + Fore.RESET)
                n_improved = 0.000000000000001
                n_not_improved = 0.00000000000000001
                force_update = False
                random_update = False
            else:
                # print(Fore.BLACK + "NO improve" + Fore.RESET)
                n_not_improved += 1

    def random_update(self):
        random_network = Network(NETWORK_SHAPE)
        for i in range(len(self.layers)):
            weights_change = random_network.layers[i].weights
            biases_change = random_network.layers[i].bias
            self.layers[i].weights += weights_change
            self.layers[i].bias += biases_change

# ----------------------Test----------------------


def test():
    data = cdp.create_data(1000)
    pred_data = data.copy()
    print(Fore.RED + f"label: {data[:, 2]}")
    input = data[:, (0, 1)]
    print(Fore.RED + f"data: {data[:, (0, 1)]}")
    cdp.plot_data(data, "True label")

    label = copy.deepcopy(data[:, 2])
    net = Network(NETWORK_SHAPE)
    # net.train()
    output = net.forward(input)
    pred = output[-1]
    class_pred = classify(pred)
    print(Fore.YELLOW + f"classify: {class_pred}")

    loss = precise_loss(pred, label)
    print(Fore.BLUE + f"loss: {loss}")

    # --------------------------- Test demand matrix ---------------------------------------------------------
    demand_matrix = get_last_layer_preAct_demands(pred, label)
    print(Fore.BLACK + "----------------------")
    print(Fore.GREEN + f"demand_matrix: {demand_matrix}")

    # --------------------------- Test adjust matrix ---------------------------------------------------------
    adjust_matrix = net.layers[-1].get_weight_adjust_matrix(output[-2], demand_matrix)
    print(Fore.BLACK + "----------------------")
    print(Fore.RED + f"adjust_matrix: {adjust_matrix}")

    # ------------------------- Test layer backward --------------------------------
    layer_backward = net.layers[-1].layer_backward(output[-2], demand_matrix)

    # --------------------------------- Test every_layers output --------------------------------------
    for output_, i in zip(output, np.arange(len(output))):
        if i == 0:
            print(Fore.RED + f"input: {output_}")
            print("----------------------")
        else:
            print(Fore.GREEN + F"layer{i}_result:{output_}")
        print("----------------------")

    # ---------------------print Net----------------------------
    for layer, i in zip(net.layers, np.arange(len(net.layers))):
        print(Fore.YELLOW + f"layer{i}_weight: {layer.weights}")
        print("----------------------")

    # ------------------------------------ Compare backward and no_backward ------------------------
    clssification = classify(output[-1])
    pred_data[:, 2] = clssification
    cdp.plot_data(pred_data, "predict_without_backward_adjust")

    backup_network = net.network_backward(output, label)
    new_outputs = backup_network.forward(input)
    new_classification = classify(new_outputs[-1])
    pred_data[:, 2] = new_classification
    cdp.plot_data(pred_data, "predict_with_backward_adjust")

    # ---------------------------------Test one batch train----------------------------------------------
    net.one_batch_train(data)
    output = net.forward(input)
    clssification = classify(output[-1])
    pred_data[:, 2] = clssification
    cdp.plot_data(pred_data, "predict_with_one_batch_train")

    # ---------------------------------Test train ---------------------
    net.train()
    output = net.forward(input)
    clssification = classify(output[-1])
    pred_data[:, 2] = clssification
    cdp.plot_data(pred_data, "After_train")

if __name__ == "__main__":
    data = cdp.create_data(1000)
    pred_data = data.copy()
    input = data[:, (0, 1)]
    cdp.plot_data(pred_data, "True label")
    net = Network(NETWORK_SHAPE)
    net.train()
    output = net.forward(input)
    clssification = classify(output[-1])
    pred_data[:, 2] = clssification
    cdp.plot_data(pred_data, "After_train")