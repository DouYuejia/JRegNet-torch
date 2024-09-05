# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.compat.v1.disable_eager_execution()

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))

# 将 base 文件夹的路径添加到 sys.path
base_path = os.path.join(project_root, 'regretNet\\base')
sys.path.append(base_path)

# 打印 sys.path 以验证路径
print(sys.path)

# 现在可以导入 base_net
from base.base_net import *


class Net(BaseNet):

    def __init__(self, config):
        super(Net, self).__init__(config)
        self.build_net()

    # def normalize_tensor(x):
    #     """
    #     该函数使用
    #     tf.nn.moments
    #     函数计算输入张量x的均值和方差，然后将其归一化。其中，axes参数指定计算均值和方差的维度，keepdims参数保持维度不变。
    #     最后，将输入张量减去均值并除以标准差（方差的平方根）即可完成归一化。注意，为了避免除以零，我们在方差上加上一个小的常数（例如1e-8）。
    #     """
    #
    #     mean, variance = tf.nn.moments(x, axes=[0, 1], keepdims=True)
    #     return (x - mean) / tf.sqrt(variance + 1e-8)

    def build_net(self):
        """
        Initializes network variables
        """
        # num_agents = self.config.num_agents
        # num_b = self.config.num_b
        # num_items = self.config.num_items
        self.num_ad = self.config.num_ad
        self.num_org = self.config.num_org
        self.num_bid = self.config.num_bid
        self.num_vol = self.config.num_vol
        self.num_agent = self.config.num_agent
        self.num_item = self.config.num_item
        self.num_bundle = self.config.num_bundle

        num_a_layers = self.config.net.num_a_layers
        num_p_layers = self.config.net.num_p_layers

        num_a_hidden_units = self.config.net.num_a_hidden_units
        num_p_hidden_units = self.config.net.num_p_hidden_units
        
                
        w_init = self.init
        b_init = tf.keras.initializers.Zeros()

        wd = None if "wd" not in self.config.train else self.config.train.wd
            
        # Alloc network weights and biases
        self.w_a = []
        self.b_a = []


        # Pay network weights and biases
        self.w_p = []
        self.b_p = []

        with tf.variable_scope("alloc"):
            # Input Layer
            self.w_a.append(create_var("w_a_0", [self.num_agent * self.num_item + self.num_bid * self.num_vol, num_a_hidden_units], initializer = w_init, wd = wd))

            # Hidden Layers
            for i in range(1, num_a_layers - 1):
                wname = "w_a_" + str(i)
                self.w_a.append(create_var(wname, [num_a_hidden_units, num_a_hidden_units], initializer = w_init, wd = wd))
                
            # Output Layer
            wname = "w_a_" + str(num_a_layers - 1)   
            self.w_a.append(create_var(wname, [num_a_hidden_units, (self.num_bundle + 1) * (self.num_item + 1)], initializer = w_init, wd = wd))

            wname = "wi_a_" + str(num_a_layers - 1)
            self.wi_a = create_var(wname, [num_a_hidden_units, (self.num_bundle + 1) * (self.num_item + 1)], initializer = w_init,  wd = wd)

            # Biases
            for i in range(num_a_layers - 1):
                wname = "b_a_" + str(i)
                self.b_a.append(create_var(wname, [num_a_hidden_units], initializer = b_init))
                
            wname = "b_a_" + str(num_a_layers - 1)   
            self.b_a.append(create_var(wname, [(self.num_bundle + 1) * (self.num_item + 1)], initializer = b_init))

            wname = "bi_a_" + str(num_a_layers - 1)
            self.bi_a = create_var(wname, [(self.num_bundle + 1) * (self.num_item + 1)], initializer = b_init)




        with tf.variable_scope("pay"):
            # Input Layer
            # 输入层只与报价有关
            self.w_p.append(create_var("w_p_0", [self.num_bid * self.num_item + self.num_bid * self.num_vol, num_p_hidden_units], initializer = w_init, wd = wd))

            # Hidden Layers
            for i in range(1, num_p_layers - 1):
                wname = "w_p_" + str(i)
                self.w_p.append(create_var(wname, [num_p_hidden_units, num_p_hidden_units], initializer = w_init, wd = wd))
                
            # Output Layer
            wname = "w_p_" + str(num_p_layers - 1)   
            self.w_p.append(create_var(wname, [num_p_hidden_units, self.num_bid], initializer = w_init, wd = wd))

            # Biases
            for i in range(num_p_layers - 1):
                wname = "b_p_" + str(i)
                self.b_p.append(create_var(wname, [num_p_hidden_units], initializer = b_init))
                
            wname = "b_p_" + str(num_p_layers - 1)   
            self.b_p.append(create_var(wname, [self.num_bid], initializer = b_init))


    def inference(self, BID, VOL, graph, bundle_item_pair):
        """
        Inference
        """
        bid_in = tf.reshape(BID, [-1, self.num_bid * self.num_item])
        vol_in = tf.reshape(VOL, [-1, self.num_vol * self.num_item])
        graph_in = tf.reshape(graph, [-1, self.num_bid * self.num_vol])
        # 分配网络的输入：报价、volume、图关系
        alloc_in = tf.concat([bid_in, vol_in, graph_in], axis=1)
        X = tf.concat([bid_in, vol_in], axis=1)
        # Allocation Network
        a = tf.matmul(alloc_in, self.w_a[0]) + self.b_a[0]  # 再议
        a = self.activation(a, 'alloc_act_0')
        activation_summary(a)

        for i in range(1, self.config.net.num_a_layers - 1):
            a = tf.matmul(a, self.w_a[i]) + self.b_a[i]
            a = self.activation(a, 'alloc_act_' + str(i))
            activation_summary(a)

        agent = tf.matmul(a, self.w_a[-1]) + self.b_a[-1]
        agent = tf.reshape(agent, [-1, self.config.num_bundle+1, (self.config.num_item+1)])

        # 用不用再议
        item = tf.matmul(a, self.wi_a) + self.bi_a
        item = tf.reshape(item, [-1, self.config.num_bundle+1, (self.config.num_item+1)])

        agent = tf.nn.softmax(agent, axis=1)
        item = tf.nn.softmax(item, axis=-1)

        #更改归一化手段
        # agent = tf.layers.batch_normalization(agent, axis=1)
        # item = tf.layers.batch_normalization(agent, axis=-1)

        az = tf.minimum(agent, item)
        
        az = tf.slice(az, [0,0,0], size=[-1, self.config.num_bundle, self.config.num_item], name = 'alloc_out')   #去零注释掉

        activation_summary(az)

        az = tf.reshape(az, [-1, self.config.num_bundle * self.config.num_item])

        azz = tf.einsum('ij,ijk->ik', az, bundle_item_pair)

        azz = tf.reshape(azz, [-1, self.config.num_agent, self.config.num_item])

        az = tf.reshape(az, [-1, self.config.num_bundle, self.config.num_item])

        a = az

        # Payment Network
        # p要不要和vol建立关系？
        # print("bid_in shape: ", bid_in.shape)
        # print("graph_in shape: ", graph_in.shape)
        # 支付网络的输入：报价、图关系
        payment_in = tf.concat([bid_in, graph_in], axis=1)
        # print("payment_in shape: ", payment_in.shape)
        # print("self.w_p[0] shape: ", self.w_p[0].shape)

        p = tf.matmul(payment_in, self.w_p[0]) + self.b_p[0]
        p = self.activation(p, 'pay_act_0')
        activation_summary(p)

        for i in range(1, self.config.net.num_p_layers - 1):
            p = tf.matmul(p, self.w_p[i]) + self.b_p[i]
            p = self.activation(p, 'pay_act_' + str(i))
            activation_summary(p)

        p = tf.matmul(p, self.w_p[-1]) + self.b_p[-1]
        p_raw = p
        p = tf.sigmoid(p, 'pay_sigmoid')
        activation_summary(p)
        

        u = tf.reduce_sum(a * tf.reshape(bid_in, [-1, self.config.num_bid, self.config.num_item]), axis = -1)

        p_tilde = p
        p = p * u
        activation_summary(p)

        social_welfare = tf.reduce_mean(tf.reduce_sum(u, axis=-1))

        return a, p, az, p_tilde, social_welfare, p_raw