# -*- coding: utf-8 -*-
import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Output-dir to write log-files and save model
__C.dir_name = os.path.join("experiments", "aaa")

__C.max_dir_name = os.path.join("max_experiments", "aaa")

# Auction params

__C.num_i =3  # 药店数量
__C.num_j =3  # 药的数量

# __C.num_agents = 4   # __C.num_agents等于__C.num_i加__C.num_j
# __C.num_items = 2
__C.num_agents = 6  # __C.num_agents等于__C.num_i加__C.num_j
__C.num_items = 3

# __C.CTR=np.array([0.6,0.35,0.05])  #长度或维度等于物品数量

__C.CTR = np.array([0.6, 0.3, 0.2])  # 长度或维度等于物品数量

__C.linjie0 =np.array([[1, 2, 3],
                       [4, 0, 5],
                       [0, 6, 0]])  # 输入邻接矩阵
__C.num_b=6
__C.distribution_type = "uniform"
__C.agent_type = "additive"     #再思考

# Save data for restore.
__C.save_data = False

# Neural Net parameters
__C.net = edict()
# initialization g - glorot, h - he + u - uniform, n - normal [gu, gn, hu, hn]
__C.net.init = "gu"
# __C.net.init = "hu"
# activations ["tanh", "sigmoid", "relu"]

# __C.net.activation = "tanh"
__C.net.activation = "tanh"

# num_a_layers, num_p_layers - total number of hidden_layers + output_layer, [a - alloc, p - pay]
# __C.net.num_a_layers = 3
# __C.net.num_p_layers = 3

__C.net.num_a_layers = 6
__C.net.num_p_layers = 6

# num_p_hidden_units, num_p_hidden_units - number of hidden units, [a - alloc, p - pay]
__C.net.num_p_hidden_units = 100
__C.net.num_a_hidden_units = 100

# __C.net.num_p_hidden_units = 100
# __C.net.num_a_hidden_units = 100

# Train paramters
__C.train = edict()
__C.train.seed = 42 # Random seed

# Iter from which training begins. If restore_iter = 0 for default. restore_iter > 0 for starting
# training form restore_iter [needs saved model]
__C.train.restore_iter = 0
# max iters to train

# __C.train.max_iter = 10000   #迭代次数
# __C.train.update_frequency = 100  #更新lanmuda的频率
# __C.train.up_op_frequency = 100000 #更新rou的频率
# __C.train.save_iter = 50000   #保存频率
# __C.train.print_iter = 100   #训练数据打印频率
# __C.val.print_iter = 10000  #测试数据打印频率


# __C.train.max_iter = 400000   #迭代次数
# __C.train.update_frequency = 100  #更新lanmuda的频率
# __C.train.up_op_frequency = 100000 #更新rou的频率
# __C.train.save_iter = 50000   #保存频率
# __C.train.print_iter = 1000   #训练数据打印频率
# __C.val.print_iter = 10000  #测试数据打印频率




# Learning rate of network param updates
__C.train.learning_rate = 1e-3
# Regularization
__C.train.wd = None

""" Train-data params """
# Choose between fixed and online. If online, set adv_reuse to False
__C.train.data = "fixed"
# Number of batches
__C.train.num_batches = 5000
# Train batch size
__C.train.batch_size = 128


""" Train-misreport params """
# Cache-misreports after misreport optimization
__C.train.adv_reuse = True
# Number of misreport initialization for training
__C.train.num_misreports = 1
# Number of steps for misreport computation
__C.train.gd_iter = 25
# Learning rate of misreport computation

__C.train.gd_lr = 0.1
# __C.train.gd_lr = 1

""" Lagrange Optimization params """
# Initial update rate
# __C.train.update_rate = 1.0

__C.train.update_rate = 0.25
__C.train.update_rate_1 = 0.25
__C.train.update_rate_2 = 0.25
# Initial Lagrange weights

# __C.train.w_rgt_init_val = 5.0

__C.train.w_rgt_init_val = 1.0

__C.train.w_rgt_init_val_1 = 1.0
__C.train.w_rgt_init_val_1_2 = 1.0
__C.train.w_rgt_init_val_1_3 = 1.0
__C.train.w_rgt_init_val_2 = 1.0
__C.train.w_rgt_init_val_3 = 1.0
__C.train.w_rgt_init_val_4 = 1.0
# Lagrange update frequency

#__C.train.update_frequency = 100

# Value by which update rate is incremented
__C.train.up_op_add = 0.25

# __C.train.up_op_add = 0.25
__C.train.up_op_add_1 = 0.25
__C.train.up_op_add_2 = 0.25
# Frequency at which update rate is incremented

#__C.train.up_op_frequency = 100000


""" train summary and save params"""
# Number of models to store on disk
__C.train.max_to_keep = 10
# Frequency at which models are saved

#__C.train.save_iter = 50000

# Train stats print frequency

#__C.train.print_iter = 1000

#__C.train.print_iter = 1


""" Validation params """
__C.val = edict()
# Number of steps for misreport computation
__C.val.gd_iter = 2000
# Learning rate for misreport computation
__C.val.gd_lr = 0.1
# __C.val.gd_lr = 1
# Number of validation batches
__C.val.num_batches = 20
# Frequency at which validation is performed

#__C.val.print_iter = 10000

#__C.val.print_iter = 1


#__C.train.max_iter = 10000   #迭代次数

# __C.train.max_iter = 2   #迭代次数
# __C.train.max_iter = 200000  #迭代次数
# __C.train.max_iter = 100000  #迭代次数
# __C.train.max_iter = 200000  #迭代次数
# __C.train.max_iter = 600000  #迭代次数
__C.train.max_iter = 200000  #迭代次数
__C.train.update_frequency = 100  #更新lanmuda的频率
# __C.train.up_op_frequency = 100000 #更新rou的频率
__C.train.up_op_frequency = 10000 #更新rou的频率
__C.train.save_iter = 50000   #模型保存频率
# __C.train.yuzhi = 0.003
__C.train.yuzhi = 0.001
# __C.train.yuzhi = 100
# __C.train.print_iter = 1000  #训练数据打印频率
# __C.train.print_iter = 1000  #训练数据打印频率

__C.train.print_iter = 1000 #训练数据打印频率
# __C.train.print_iter = 1 #训练数据打印频率

# __C.val.print_iter = 10000  #测试数据打印频率

# __C.val.print_iter = 750  #测试数据打印频率
# __C.val.print_iter = 10000  #测试数据打印频率

__C.val.print_iter =10000 #测试数据打印频率
# __C.val.print_iter =1 #测试数据打印频率

# __C.val.print_iter = 750  #测试数据打印频率

# __C.train.max_iter = 400000   #迭代次数
# __C.train.update_frequency = 100  #更新lanmuda的频率
# __C.train.up_op_frequency = 100000 #更新rou的频率
# __C.train.save_iter = 50000   #保存频率
# __C.train.print_iter = 1000   #训练数据打印频率
# __C.val.print_iter = 10000  #测试数据打印频率

# Validation data frequency
__C.val.data = "fixed"
# __C.val.data = "online"

""" Test params """
# Test set
__C.test = edict()
# Test Seed
__C.test.seed = 100
# Model to be evaluated
__C.test.restore_iter = 30000
# Number of misreports

# __C.test.num_misreports = 1000

# __C.test.num_misreports = 3
__C.test.num_misreports = 1

# Number of steps for misreport computation

# __C.test.gd_iter = 2000
__C.test.gd_iter = 50

# __C.test.gd_iter = 100

# __C.test.gd_iter = 20

# Learning rate for misreport computation
__C.test.gd_lr = 0.1
# Test data
# __C.test.data = "online"
__C.test.data = "fixed"
# Number of test batches

__C.test.num_batches = 100
# __C.test.num_batches = 3

# Test batch size
# __C.test.batch_size = 100
__C.test.batch_size = 128
# Save Ouput
__C.test.save_output = False


# Fixed Val params
__C.val.batch_size = __C.train.batch_size
__C.val.num_misreports = __C.train.num_misreports

# Compute number of samples
__C.train.num_instances = __C.train.num_batches * __C.train.batch_size
__C.val.num_instances = __C.val.num_batches * __C.val.batch_size
__C.test.num_instances = __C.test.num_batches * __C.test.batch_size

