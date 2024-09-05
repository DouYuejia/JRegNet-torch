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

__C.num_i =10  # 药店数量
__C.num_j =15  # 药的数量

# __C.num_agents = 4   # __C.num_agents等于__C.num_i加__C.num_j
# __C.num_items = 2
__C.num_agents = 25  # __C.num_agents等于__C.num_i加__C.num_j
__C.num_items = 20



# __C.CTR=np.array([0.6,0.35,0.05])  #长度或维度等于物品数量

# __C.CTR = np.array([0.9993605886774073, 0.9956716132990344, 0.9856642777657987, 0.9743611659965519, 0.9707706646355734, 0.9677575243121731, 0.967364875853909, 0.9562858834493909, 0.9499620656905613, 0.9259135648673955, 0.9045087374151454, 0.9018345036268559, 0.8727011958838873, 0.8588345867379021, 0.850813196250152, 0.8478957340651234, 0.7924961008272824, 0.7892466856265373, 0.7814911428699287, 0.7529067892033792, 0.7076168721296627, 0.6950218867227409, 0.694850744824444, 0.6820361341262202, 0.5881782313078352, 0.545396109662423, 0.5424317016826081, 0.5420527540019768, 0.48709329668078016, 0.46824115230744934, 0.45547790024178425, 0.4506318366222378, 0.4426001476935202, 0.4129017986765291, 0.36072097222659905, 0.30424223072216416, 0.2873111323992116, 0.2792674598656114, 0.2759217524452109, 0.2600597071122466, 0.24362003128050502, 0.19256564653076258, 0.18251457698226747, 0.16688382577133776, 0.11312892098181937, 0.11310693091761115, 0.08898124018665043, 0.0792588143647891, 0.05735444290331837, 0.05051043995213844])  # 长度或维度等于物品数量
__C.CTR = np.array([0.9664930694094844, 0.8744945705415402, 0.7025216472010907, 0.6627377156526145, 0.6423090974764666, 0.5973762621142088, 0.5199552647774152, 0.50568828305045, 0.48387610745976595, 0.4748163491521369, 0.4453359484481987, 0.3872460752899852, 0.35829432446163034, 0.3525196533694087, 0.2563634040743503, 0.22549344600256138, 0.22436560843507047, 0.11208884465832669, 0.10229428109757355, 0.02279242497597289])

__C.linjie0 =np.array([[1,0,0,2,0,3,4,0,5,0,6,0,7,0,8],
                       [9,0,10,11,0,0,0,0,0,0,12,0,13,14,0],
                       [0,0,0,0,15,0,16,0,0,17,0,0,18,0,0],
                       [19,0,0,0,0,0,0,20,0,0,0,21,0,0,22],
                       [0,0,0,0,0,0,0,0,0,23,24,0,25,26,0],
                       [0,27,28,0,29,30,0,31,0,32,0,33,0,34,35],
                       [36,0,37,0,0,0,0,38,0,0,0,0,39,40,41],
                       [42,0,0,0,0,0,0,0,0,0,0,0,0,0,43],
                       [0,0,44,0,0,45,0,0,46,0,0,47,0,0,0],
                       [0,0,0,48,0,0,0,49,0,0,50,0,0,0,0]])  # 输入邻接矩阵
__C.num_b=50
# __C.config.bianshu=50

__C.diedai_serch_max=0
__C.diedai_serch_min=100

__C.distribution_type = "uniform"
# __C.agent_type = "additive"     #再思考
__C.agent_type = "unit_demand"


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
# __C.train.seed = 52 # Random seed
# __C.train.seed = 62 # Random seed
# __C.train.seed = 72 # Random seed
# __C.train.seed = 82 # Random seed
# __C.train.seed = 92 # Random seed

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
# __C.train.up_op_add = 1.0

__C.train.up_op_add = 0.25
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
# __C.train.max_iter = 300000  #迭代次数
__C.train.max_iter = 300000  #迭代次数
# __C.train.max_iter = 130000  #迭代次数
# __C.train.max_iter = 150000  #迭代次数
# __C.train.max_iter = 110000  #迭代次数
# __C.train.max_iter = 90000  #迭代次数
# __C.train.max_iter = 2  #迭代次数
__C.train.update_frequency = 100  #更新lanmuda的频率
# __C.train.up_op_frequency = 100000 #更新rou的频率
__C.train.up_op_frequency = 10000 #更新rou的频率
__C.train.save_iter = 50000   #模型保存频率
__C.train.yuzhi = 0.001
# __C.train.yuzhi = 100
__C.train.print_iter = 1000  #训练数据打印频率
# __C.train.print_iter = 1  #训练数据打印频率

# __C.val.print_iter = 10000  #测试数据打印频率

__C.val.print_iter = 10000  #测试数据打印频率
# __C.val.print_iter = 1  #测试数据打印频率

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
# __C.test.seed = 110
# __C.test.seed = 120
# __C.test.seed = 130
# __C.test.seed = 140
# __C.test.seed = 150

# Model to be evaluated
__C.test.restore_iter = 30000
# Number of misreports

# __C.test.num_misreports = 1000

__C.test.num_misreports = 3
# __C.test.num_misreports = 1

# Number of steps for misreport computation

# __C.test.gd_iter = 2000
__C.test.gd_iter = 50

# __C.test.gd_iter = 100

# __C.test.gd_iter = 20

# Learning rate for misreport computation
__C.test.gd_lr = 0.1
# Test data
__C.test.data = "online"
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

