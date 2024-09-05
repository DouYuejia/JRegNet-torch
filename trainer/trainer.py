# -*- coding: utf-8 -*-



import os
import sys
import time
import logging
import numpy as np

import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.compat.v1.disable_eager_execution()



class Trainer(object):

    def __init__(self, config, mode, net, clip_op_lambda):
        self.config = config
        self.mode = mode
        
        # 创建输出目录
        if not os.path.exists(self.config.dir_name):
            os.makedirs(self.config.dir_name)

        if not os.path.exists(self.config.max_dir_name):

            os.makedirs(self.config.max_dir_name)
            #os.mkdir(self.config.dir_name)

        if self.mode == "train":
            log_suffix = '_' + str(self.config.num_ad) + '*ads_' + str(self.config.num_org) + '*orgs_' + str(self.config.CTR.tolist()) + '_CTR_'
            log_suffix = '_' + str(self.config.train.restore_iter) if self.config.train.restore_iter > 0 else ''
            self.log_fname = os.path.join(self.config.dir_name, 'train' + log_suffix + '.txt')
        else:
            log_suffix = "_iter_" + str(self.config.test.restore_iter) + "_m_" + str(self.config.test.num_misreports) + "_gd_" + str(self.config.test.gd_iter)
            self.log_fname = os.path.join(self.config.dir_name, "test" + log_suffix + ".txt")
            
        # 设置随机种子，便于复现
        np.random.seed(self.config[self.mode].seed)
        tf.set_random_seed(self.config[self.mode].seed)
        
        # 初始化 Logger
        self.init_logger()

        # 初始化网络
        self.net = net
        
        ## 对lambda进行clip操作
        self.clip_op_lambda = clip_op_lambda
        
        # 初始化tf图
        self.init_graph()
        
    def get_clip_op(self, adv_var):
        self.clip_op =  self.clip_op_lambda(adv_var)
        #tf.assign(adv_var, tf.clip_by_value(adv_var, 0.0, 1.0))

    def get_clip_op_test(self, adv_var_test):
        self.clip_op_test = self.clip_op_lambda(adv_var_test)
        

    def init_logger(self):
        '''
        初始化logger
        '''

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        handler = logging.FileHandler(self.log_fname, 'w')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        self.logger = logger

    def compute_rev(self, pay):
        """
            根据payment计算revenue
            Given payment (pay), computes revenue
            Input params:
                pay: [num_batches, num_agents]
            Output params:
                revenue: scalar
        """
        return tf.reduce_mean(tf.reduce_sum(pay, axis=-1))

    def compute_utility(self, x, alloc, pay):
        """
            根据分配结果、输入的value值与payment值计算utility
            Given input valuation (x), payment (pay) and allocation (alloc), computes utility
            Input params:
                x: [num_batches, num_agents, num_items]
                a: [num_batches, num_agents, num_items]
                p: [num_batches, num_agents]
            Output params:
                utility: [num_batches, num_agents]
        """
        return tf.reduce_sum(tf.multiply(alloc, x), axis=-1) - pay


    def get_misreports(self, ad, org, vol, adv_var, adv_var_CTR, adv_shape, graph_raw , bundle_item_pair):
        """
        计算误报值
        没太搞懂为什么要生成2次adv_var然后再相乘
        Args:
            ad: [num_batches, num_ad, num_item]
            org: [num_batched, num_org, num_item]
            vol: [num_batched, num_vol, num_item]
            adv_var: [num_misreports, batch_size, num_agents, 1]
            adv_var_CTR: [num_misreports, batch_size, num_agents, num_item]
            adv_shape: [num_ad, num_misreports, batch_size, num_agents, num_item]
            graph_raw: [batch_size, num_bid, num_vol]
            bundle_item_pair: [batch_size, num_bundle * num_item, num_agent * num_item]

        Returns:
            ad_mis, mis_org, mis_vol, mis_bid, misreports, mis_graph, mis_pair
        """
        bid = tf.concat([ad, org], axis=1)
        num_misreports = adv_shape[1]
        adv_var = tf.tile(adv_var, [1, 1, 1, self.config.num_item])
        adv_var = adv_var * adv_var_CTR
        adv = tf.tile(tf.expand_dims(adv_var, 0), [self.config.num_bid, 1, 1, 1, 1])
        mis_bid = tf.tile(bid, [self.config.num_bid * num_misreports, 1, 1])
        # mis_org = tf.tile(org, [self.config.num_bid * num_misreports, 1, 1])
        mis_vol = tf.tile(vol, [self.config.num_bid * num_misreports, 1, 1])
        # mis_bid = tf.concat([ad_mis, mis_org], axis=1)

        ad_r = tf.reshape(mis_bid, adv_shape)
        
        y = ad_r * (1 - self.adv_mask) + adv * self.adv_mask
        misreports = tf.reshape(y, [-1, self.config.num_bid, self.config.num_item])

        mis_graph = tf.tile(graph_raw, [adv_shape[0] * adv_shape[1], 1, 1])
        mis_pair = tf.tile(bundle_item_pair, [adv_shape[0] * adv_shape[1], 1, 1])
        return mis_bid, mis_vol, misreports, mis_graph, mis_pair


    def init_graph(self):
        """
        初始化训练图
        Returns:

        """
        ad_shape = [self.config[self.mode].batch_size, self.config.num_ad, self.config.num_item]
        org_shape = [self.config[self.mode].batch_size, self.config.num_org, self.config.num_item]
        bid_shape = [self.config[self.mode].batch_size, self.config.num_bid, self.config.num_item]
        vol_shape = [self.config[self.mode].batch_size, self.config.num_vol, self.config.num_item]
        agent_shape = [self.config[self.mode].batch_size, self.config.num_agent, self.config.num_item]
        graph_raw_shape = [self.config[self.mode].batch_size, self.config.num_bid, self.config.num_vol]
        graph_ranked_shape = [self.config[self.mode].batch_size, self.config.num_bid, self.config.num_vol]
        bundle_item_pair_shape = [self.config[self.mode].batch_size, self.config.num_bundle * self.config.num_item, self.config.num_agent * self.config.num_item]
        adv_shape = [self.config.num_bid, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_bid, self.config.num_item]
        adv_shape_test = [self.config.num_bid, self.config.test.num_misreports, self.config.test.batch_size, self.config.num_bid, self.config.num_item]
        adv_var_shape = [self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_bid, 1]
        adv_var_CTR_shape = [self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_bid, self.config.num_item]
        adv_var_shape_test = [self.config.test.num_misreports, self.config.test.batch_size, self.config.num_bid, 1]
        adv_var_CTR_shape_test = [self.config.test.num_misreports, self.config.test.batch_size, self.config.num_bid, self.config.num_item]
        u_shape = [self.config.num_bid, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_bid]
        u_shape_test = [self.config.num_bid, self.config.test.num_misreports, self.config.test.batch_size, self.config.num_bid]

        # Placeholders
        self.ad = tf.placeholder(tf.float32, shape=ad_shape, name='ad')
        self.org = tf.placeholder(tf.float32, shape=org_shape, name='org')
        self.bid = tf.placeholder(tf.float32, shape=bid_shape, name='bid')
        self.vol = tf.placeholder(tf.float32, shape=vol_shape, name='vol')
        self.agent = tf.placeholder(tf.float32, shape=agent_shape, name='agent')
        self.graph_raw = tf.placeholder(tf.float32, shape=graph_raw_shape, name='graph_raw')
        self.graph_ranked = tf.placeholder(tf.float32, shape=graph_ranked_shape, name='graph_ranked')
        self.bundle_item_pair = tf.placeholder(tf.float32, shape=bundle_item_pair_shape, name='bundle_item_pair')
        self.adv_init = tf.placeholder(tf.float32, shape=adv_var_shape, name='adv_init')
        self.adv_init_CTR = tf.placeholder(tf.float32, shape=adv_var_CTR_shape, name='adv_init_CTR')
        self.adv_init_test = tf.placeholder(tf.float32, shape=adv_var_shape_test, name='adv_init_test')
        self.adv_init_CTR_test = tf.placeholder(tf.float32, shape=adv_var_CTR_shape_test, name='adv_init_CTR_test')

        self.adv_mask = np.zeros(adv_shape)
        self.adv_mask[np.arange(self.config.num_bid), :, :, np.arange(self.config.num_bid), :] = 1.0

        self.adv_mask_test = np.zeros(adv_shape_test)
        self.adv_mask_test[np.arange(self.config.num_bid), :, :, np.arange(self.config.num_bid), :] = 1.0
        
        self.u_mask = np.zeros(u_shape)
        self.u_mask_test = np.zeros(u_shape_test)
        self.u_mask[np.arange(self.config.num_bid), :, :, np.arange(self.config.num_bid)] = 1.0
        self.u_mask_test[np.arange(self.config.num_bid), :, :, np.arange(self.config.num_bid)] = 1.0

        with tf.variable_scope('adv_var'):
            self.adv_var = tf.get_variable('adv_var', shape = adv_var_shape, dtype = tf.float32)

        with tf.variable_scope('adv_var_CTR'):
            self.adv_var_CTR = tf.get_variable('adv_var_CTR', shape = adv_var_CTR_shape, dtype = tf.float32)

        with tf.variable_scope('adv_var_test'):
            self.adv_var_test = tf.get_variable('adv_var_test', shape = adv_var_shape_test, dtype = tf.float32)

        with tf.variable_scope('adv_var_test_CTR'):
            self.adv_var_CTR_test = tf.get_variable('adv_var_CTR_test', shape = adv_var_CTR_shape_test, dtype = tf.float32)


        # 获取误报值
        self.mis_bid, self.mis_vol, self.misreports,self.mis_graph, self.mis_pair =\
            self.get_misreports(self.ad, self.org, self.vol, self.adv_var, self.adv_var_CTR, adv_shape, self.graph_raw, self.bundle_item_pair)
        self.mis_bid_test, self.mis_vol_test, self.misreports_test,self.mis_graph_test, self.mis_pair_test =\
            self.get_misreports(self.ad, self.org, self.vol,  self.adv_var_test, self.adv_var_CTR_test, adv_shape_test, self.graph_raw, self.bundle_item_pair)
        #
        self.bid = tf.concat([self.ad, self.org], axis=1)
        # 获取机制在真实报价下的分配、支付、bundle、p_tilde、社会福利值
        self.alloc, self.pay, self.bundle, self.p_tilde, self.social_welfare, self.p_raw = \
            self.net.inference(self.bid, self.vol, self.graph_raw, self.bundle_item_pair)

        print("self.mis_bid shape:", self.mis_bid.shape)
        print("self.misreports shape:", self.misreports.shape)
        # print("self.mis_org shape:", self.mis_org.shape)
        # self.mis_ad_org = tf.concat([self.misreports, self.mis_org], axis=1)
        # self.mis_ad_org_test = tf.concat([self.misreports_test, self.mis_org_test], axis=1)
        

        print("self.mis_pair shape: ", self.mis_pair.shape)
        self.a_mis, self.p_mis, self.mis_bundle, self.mis_p_tilde, _, self.p_raw_mis = self.net.inference(self.misreports, self.mis_vol, self.mis_graph, self.mis_pair)
        self.a_mis_test, self.p_mis_test, self.mis_bundle_test, self.mis_p_tilde_test, _, self.p_raw_mis_test = self.net.inference(self.misreports_test, self.mis_vol_test, self.mis_graph_test, self.mis_pair_test)

        # 计算机制得出的utility值
        # print("self.alloc shape: ", self.alloc.shape)
        # print("self.pay shape: ", self.pay)
        self.utility = self.compute_utility(self.bid, self.alloc[:,:bid_shape[1],:], self.pay[:,:bid_shape[1]])
        self.utility_mis = self.compute_utility(self.mis_bid, self.a_mis[:,:bid_shape[1],:], self.p_mis[:,:bid_shape[1]])
        self.utility_mis_test = self.compute_utility(self.mis_bid_test, self.a_mis_test[:,:bid_shape[1],:], self.p_mis_test[:,:bid_shape[1]])

        # 计算机制的regret值
        # print("self.utility_mis shape: ", self.utility_mis.shape)
        # print("u_shape: ", u_shape)
        u_mis = tf.reshape(self.utility_mis, u_shape) * self.u_mask
        u_mis_test = tf.reshape(self.utility_mis_test, u_shape_test) * self.u_mask_test
        utility_true = tf.tile(self.utility, [self.config.num_bid * self.config[self.mode].num_misreports, 1])
        self.utility_true = utility_true
        utility_true_test = tf.tile(self.utility, [self.config.num_bid * self.config.test.num_misreports, 1])
        excess_from_utility = tf.nn.relu(tf.reshape(self.utility_mis - utility_true, u_shape) * self.u_mask)
        self.excess_from_utility = excess_from_utility
        excess_from_utility_test = tf.nn.relu(tf.reshape(self.utility_mis_test - utility_true_test, u_shape_test) * self.u_mask_test)


        rgt = tf.reduce_mean(tf.reduce_max(excess_from_utility, axis=(1, 3)), axis=1)

        rgt_test = tf.reduce_mean(tf.reduce_max(excess_from_utility_test, axis=(1, 3)), axis=1)

        # 评估指标
        revenue = self.compute_rev(self.pay)
        self.revenue = self.compute_rev(self.pay)

        social_welfare = self.social_welfare


        rgt_mean = tf.reduce_mean(rgt)
        rgt_mean_test = tf.reduce_mean(rgt_test)
        self.rgt= tf.reduce_mean(rgt)
        irp_mean = tf.reduce_mean(tf.nn.relu(-self.utility))

        # 变量列表
        alloc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='alloc')
        pay_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pay')
        var_list = alloc_vars + pay_vars


        
        if self.mode is "train":

            w_rgt_init_val = 0.0 if "w_rgt_init_val" not in self.config.train else self.config.train.w_rgt_init_val

            with tf.variable_scope('lag_var'):
                self.w_rgt = tf.Variable(np.ones(self.config.num_bid).astype(np.float32) * w_rgt_init_val, 'w_rgt')

            update_rate = tf.Variable(self.config.train.update_rate, trainable = False)
            self.increment_update_rate = update_rate.assign(update_rate + self.config.train.up_op_add)
      
            # Loss Functions
            rgt_penalty = update_rate * tf.reduce_sum(tf.square(rgt)) / 2.0        
            lag_loss = tf.reduce_sum(self.w_rgt * rgt)


            winning_volumes = tf.reduce_sum(tf.multiply(self.alloc, self.vol), axis=-1)

            gmv = tf.reduce_mean(tf.reduce_sum(winning_volumes, axis=-1))

            loss_1 = -revenue - gmv + rgt_penalty + lag_loss
            loss_2 = -tf.reduce_sum(u_mis)
            loss_3 = -lag_loss
            loss_test = -tf.reduce_sum(u_mis_test)

            reg_losses = tf.get_collection('reg_losses')
            if len(reg_losses) > 0:
                reg_loss_mean = tf.reduce_mean(reg_losses)
                loss_1 = loss_1 + reg_loss_mean

             
            learning_rate = tf.Variable(self.config.train.learning_rate, trainable = False)
        
            # Optimizer
            opt_1 = tf.train.AdamOptimizer(learning_rate)
            opt_2 = tf.train.AdamOptimizer(self.config.train.gd_lr)
            opt_3 = tf.train.GradientDescentOptimizer(update_rate)


            # Train ops
            self.train_op  = opt_1.minimize(loss_1, var_list = var_list)
            self.train_mis_step = opt_2.minimize(loss_2, var_list = [self.adv_var])
            self.lagrange_update    = opt_3.minimize(loss_3, var_list = [self.w_rgt])
            
            # Val ops
            val_mis_opt = tf.train.AdamOptimizer(self.config.val.gd_lr)
            self.val_mis_step = val_mis_opt.minimize(loss_2, var_list = [self.adv_var])

            test_mis_opt = tf.train.AdamOptimizer(self.config.test.gd_lr)
            self.test_mis_step = test_mis_opt.minimize(loss_test, var_list=[self.adv_var_test])


            # Reset ops
            self.reset_train_mis_opt = tf.variables_initializer(opt_2.variables()) 
            self.reset_val_mis_opt = tf.variables_initializer(val_mis_opt.variables())
            self.reset_test_mis_opt = tf.variables_initializer(test_mis_opt.variables())

            # Metrics
            self.metrics = [revenue, gmv, rgt_mean, rgt_penalty, lag_loss, loss_1, tf.reduce_mean(self.w_rgt), update_rate]
            self.metric_names = ["Revenue", "gmv", "Regret", "Reg_Loss", "Lag_Loss", "Net_Loss", "w_rgt_mean", "update_rate"]
            
            #Summary
            tf.summary.scalar('revenue', revenue)
            tf.summary.scalar('gmv', gmv)
            tf.summary.scalar('regret', rgt_mean)
            tf.summary.scalar('reg_loss', rgt_penalty)
            tf.summary.scalar('lag_loss', lag_loss)
            tf.summary.scalar('net_loss', loss_1)
            tf.summary.scalar('w_rgt_mean', tf.reduce_mean(self.w_rgt))
            if len(reg_losses) > 0: tf.summary.scalar('reg_loss', reg_loss_mean)

            self.metrics_t = [revenue, rgt_mean_test, gmv, irp_mean, social_welfare]
            self.metric_names_t = ["Revenue", "Regret", "GMV", "IRP", "social_welfare"]

            self.merged = tf.summary.merge_all()
            self.saver = tf.train.Saver(max_to_keep = self.config.train.max_to_keep)
            self.saver1 = tf.train.Saver(max_to_keep=self.config.train.max_to_keep)
        
        elif self.mode is "test":

            loss = -tf.reduce_sum(u_mis)
            test_mis_opt = tf.train.AdamOptimizer(self.config.test.gd_lr)
            self.test_mis_step = test_mis_opt.minimize(loss, var_list = [self.adv_var])
            self.reset_test_mis_opt = tf.variables_initializer(test_mis_opt.variables())

            # Metrics
            welfare = tf.reduce_mean(tf.reduce_sum(self.alloc[:,:ad_shape[1],:] * self.ad, axis = (1,2)))
            self.metrics = [revenue, rgt_mean, gmv, irp_mean, welfare]
            self.metric_names = ["Revenue", "Regret", "GMV", "IRP", "welfare"]
            self.saver = tf.train.Saver(var_list = var_list)
            

        # Helper ops post GD steps
        self.assign_op = tf.assign(self.adv_var, self.adv_init)
        self.assign_op_CTR = tf.assign(self.adv_var_CTR, self.adv_init_CTR)
        self.assign_op_test = tf.assign(self.adv_var_test, self.adv_init_test)
        self.assign_op_CTR_test = tf.assign(self.adv_var_CTR_test, self.adv_init_CTR_test)
        self.get_clip_op(self.adv_var)
        self.get_clip_op_test(self.adv_var_test)
        
    def train(self, generator):
        """
        Runs training
        """
        
        self.train_gen, self.val_gen, self.test_gen = generator
        self.rgt_yuzhi_1 = self.config.train.yuzhi_1
        self.rgt_yuzhi = self.config.train.yuzhi
        self.max_revn= -100
        self.max_iter = -1
        self.max_rgt = -1
        self.max_model_path=""

        self.max_revn_test = -100
        self.max_iter_test = -1
        self.max_rgt_test = -1
        self.max_ir_test = -100
        self.max_social_welfare_test = -1

        iter = self.config.train.restore_iter
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(self.config.dir_name, sess.graph)

        
        if iter > 0:
            model_path = os.path.join(self.config.dir_name, 'model-' + str(iter))
            self.saver.restore(sess, model_path)

        if iter == 0:
            self.train_gen.save_data(0)
            self.saver.save(sess, os.path.join(self.config.dir_name,'model'), global_step = iter)

        time_elapsed = 0.0

        ADV_CTR = np.tile(self.config.CTR.reshape(1, 1, 1, self.config.num_item),\
                          [self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_bid, 1])


        sess.run(self.assign_op_CTR, feed_dict={self.adv_init_CTR: ADV_CTR[:,:,:self.config.num_bid,:]})

        ADV_CTR_test = np.tile(self.config.CTR.reshape(1, 1, 1, self.config.num_item),\
                          [self.config.test.num_misreports, self.config[self.mode].batch_size, self.config.num_bid, 1])



        sess.run(self.assign_op_CTR_test, feed_dict={self.adv_init_CTR_test: ADV_CTR_test[:,:,:self.config.num_bid,:]})

        while iter < (self.config.train.max_iter):
            # Get a mini-batch
            # X, ADV, ADV1, perm = next(self.train_gen.gen_func)
            AD, ORG, BID, VOL, graph_raw, graph_ranked, bundle_item_pair, ADV, perm = next(self.train_gen.gen_func)

            if iter == 0:
                sess.run(self.lagrange_update, feed_dict = {self.ad: AD, self.org: ORG, self.bid: BID, self.vol:VOL, \
                                                            self.graph_raw: graph_raw, self.graph_ranked:graph_ranked, \
                                                            self.bundle_item_pair: bundle_item_pair})
            tic = time.time()

            # 获取最优误报值
            sess.run(self.assign_op, feed_dict = {self.adv_init: ADV})
            for _ in range(self.config.train.gd_iter):
                sess.run(self.train_mis_step, feed_dict = {self.ad: AD, self.org: ORG, self.bid: BID, self.vol:VOL, \
                                                            self.graph_raw: graph_raw, self.graph_ranked:graph_ranked, \
                                                            self.bundle_item_pair: bundle_item_pair})
                sess.run(self.clip_op)
            sess.run(self.reset_train_mis_opt)

            if self.config.train.data is "fixed" and self.config.train.adv_reuse:
                self.train_gen.update_adv(perm, sess.run(self.adv_var))

            # Update network params
            sess.run(self.train_op, feed_dict = {self.ad: AD, self.org: ORG, self.bid: BID, self.vol:VOL, \
                                                 self.graph_raw: graph_raw, self.graph_ranked:graph_ranked, \
                                                 self.bundle_item_pair: bundle_item_pair})
                
            if iter==0:
                summary = sess.run(self.merged, feed_dict = {self.ad: AD, self.org: ORG, self.bid: BID, self.vol:VOL, \
                                                             self.graph_raw: graph_raw, self.graph_ranked:graph_ranked, \
                                                             self.bundle_item_pair: bundle_item_pair})
                train_writer.add_summary(summary, iter) 

            iter += 1
            # if iter % 20 == 0:
            #     p_raw = sess.run(self.p_raw, feed_dict={self.ad: AD, self.org: ORG, self.bid: BID, self.vol: VOL, \
            #                                                 self.graph_raw: graph_raw, self.graph_ranked: graph_ranked, \
            #                                                 self.bundle_item_pair: bundle_item_pair})
            #     p_raw_mis = sess.run(self.p_raw_mis, feed_dict={self.ad: AD, self.org: ORG, self.bid: BID, self.vol: VOL, \
            #                                             self.graph_raw: graph_raw, self.graph_ranked: graph_ranked, \
            #                                             self.bundle_item_pair: bundle_item_pair})
            #     p_raw_mis_test = sess.run(self.p_raw_mis_test, feed_dict={self.ad: AD, self.org: ORG, self.bid: BID, self.vol: VOL, \
            #                                             self.graph_raw: graph_raw, self.graph_ranked: graph_ranked, \
            #                                             self.bundle_item_pair: bundle_item_pair})
            #     print("p_raw: ", p_raw[:5])
            #     print("p_raw_mis: ", p_raw_mis[:5])
            #     print("p_raw_mis_test", p_raw_mis_test[:5])


            if iter % 100 == 0:
                alloc = sess.run(self.alloc, feed_dict={self.ad: AD, self.org: ORG, self.bid: BID, self.vol: VOL, \
                                                        self.graph_raw: graph_raw, self.graph_ranked: graph_ranked, \
                                                        self.bundle_item_pair: bundle_item_pair})
                pay = sess.run(self.pay, feed_dict={self.ad: AD, self.org: ORG, self.bid: BID, self.vol: VOL, \
                                                    self.graph_raw: graph_raw, self.graph_ranked: graph_ranked, \
                                                    self.bundle_item_pair: bundle_item_pair})
                bid = sess.run(self.bid, feed_dict={self.ad: AD, self.org: ORG, self.bid: BID, self.vol: VOL, \
                                                    self.graph_raw: graph_raw, self.graph_ranked: graph_ranked, \
                                                    self.bundle_item_pair: bundle_item_pair})
                ad = sess.run(self.ad, feed_dict={self.ad: AD, self.org: ORG, self.bid: BID, self.vol: VOL, \
                                                  self.graph_raw: graph_raw, self.graph_ranked: graph_ranked, \
                                                  self.bundle_item_pair: bundle_item_pair})
                org = sess.run(self.org, feed_dict={self.ad: AD, self.org: ORG, self.bid: BID, self.vol: VOL, \
                                                    self.graph_raw: graph_raw, self.graph_ranked: graph_ranked, \
                                                    self.bundle_item_pair: bundle_item_pair})
                vol = sess.run(self.vol, feed_dict={self.ad: AD, self.org: ORG, self.bid: BID, self.vol: VOL, \
                                                    self.graph_raw: graph_raw, self.graph_ranked: graph_ranked, \
                                                    self.bundle_item_pair: bundle_item_pair})
                p_tilde = sess.run(self.p_tilde, feed_dict={self.ad: AD, self.org: ORG, self.bid: BID, self.vol: VOL, \
                                                    self.graph_raw: graph_raw, self.graph_ranked: graph_ranked, \
                                                    self.bundle_item_pair: bundle_item_pair})
                social_welfare = sess.run(self.social_welfare, feed_dict={self.ad: AD, self.org: ORG, self.bid: BID, self.vol: VOL, \
                                                    self.graph_raw: graph_raw, self.graph_ranked: graph_ranked, \
                                                    self.bundle_item_pair: bundle_item_pair})
                utility = sess.run(self.utility_true, feed_dict={self.ad: AD, self.org: ORG, self.bid: BID, self.vol: VOL, \
                                                    self.graph_raw: graph_raw, self.graph_ranked: graph_ranked, \
                                                    self.bundle_item_pair: bundle_item_pair})
                utility_mis = sess.run(self.utility_mis, feed_dict={self.ad: AD, self.org: ORG, self.bid: BID, self.vol: VOL, \
                                                    self.graph_raw: graph_raw, self.graph_ranked: graph_ranked, \
                                                    self.bundle_item_pair: bundle_item_pair})
                RGT = sess.run(self.excess_from_utility, feed_dict={self.ad: AD, self.org: ORG, self.bid: BID, self.vol: VOL, \
                                                    self.graph_raw: graph_raw, self.graph_ranked: graph_ranked, self.bundle_item_pair: bundle_item_pair})
                # log_str = "TRAIN-BATCH Iter: %d, t = %.4f" % (iter,
                #                                               time_elapsed) + "\n alloc: \n%s, \n pay: \n %s, \n ad: \n %s, \n org_bid: \n %s, \n bid: \n %s, \n vol: \n %s, \n p_tilde: \n %s, \n sw: \n %s, \n utility_true: \n %s, \n utility_mis: \n %s" % (
                #           alloc[0], pay[0], ad[0], org[0], bid[0], vol[0], p_tilde[0], social_welfare, utility[0], utility_mis[0])
                log_str = "TRAIN-BATCH Iter: %d, t = %.4f" % (iter,
                                                              time_elapsed) + "\n alloc: \n%s, \n pay: \n %s, \n bid: \n %s, \n vol: \n %s, \n p_tilde: \n %s, \n sw: \n %s, " % (
                              alloc[:5], pay[:5], bid[:5], vol[:5], p_tilde[:5], social_welfare)
                # print("excess_from_utility shape: ", self.excess_from_utility.shape)
                self.logger.info(log_str)





            # Run Lagrange Update
            if iter % self.config.train.update_frequency == 0:
                sess.run(self.lagrange_update, feed_dict = {self.ad: AD, self.org: ORG, self.bid: BID, self.vol:VOL, \
                                                            self.graph_raw: graph_raw, self.graph_ranked:graph_ranked, \
                                                            self.bundle_item_pair: bundle_item_pair})
                

            if iter % self.config.train.up_op_frequency == 0:
                sess.run(self.increment_update_rate)

            toc = time.time()
            time_elapsed += (toc - tic)
                        
            if ((iter % self.config.train.save_iter) == 0) or (iter == self.config.train.max_iter):
                self.saver.save(sess, os.path.join(self.config.dir_name,'model'), global_step = iter)
                self.train_gen.save_data(iter)

            if (iter % self.config.train.print_iter) == 0:
                # Train Set Stats
                summary = sess.run(self.merged, feed_dict = {self.ad: AD, self.org: ORG, self.bid: BID, self.vol:VOL, \
                                                             self.graph_raw: graph_raw, self.graph_ranked:graph_ranked, \
                                                             self.bundle_item_pair: bundle_item_pair})
                train_writer.add_summary(summary, iter)
                metric_vals = sess.run(self.metrics, feed_dict = {self.ad: AD, self.org: ORG, self.bid: BID, self.vol:VOL, \
                                                                  self.graph_raw: graph_raw, self.graph_ranked:graph_ranked, \
                                                                  self.bundle_item_pair: bundle_item_pair})
                fmt_vals = tuple([ item for tup in zip(self.metric_names, metric_vals) for item in tup ])
                log_str = "TRAIN-BATCH Iter: %d, t = %.4f"%(iter, time_elapsed) + ", %s: %.6f"*len(self.metric_names)%fmt_vals
                self.logger.info(log_str)



            rgt_xn = sess.run(self.rgt, feed_dict={self.ad: AD, self.org: ORG, self.bid: BID, self.vol:VOL, \
                                                   self.graph_raw: graph_raw, self.graph_ranked:graph_ranked, \
                                                   self.bundle_item_pair: bundle_item_pair})
            revn_xn = sess.run(self.revenue, feed_dict={self.ad: AD, self.org: ORG, self.bid: BID, self.vol:VOL, \
                                                        self.graph_raw: graph_raw, self.graph_ranked:graph_ranked, \
                                                        self.bundle_item_pair: bundle_item_pair})

            # if ((iter % self.config.train.save_iter) == 0) or (iter == self.config.train.max_iter) or rgt_xn<self.rgt_yuzhi:

            #iter>self.config.train.max_iter-200000 100  150000
            if(iter>self.config.train.max_iter-100):
                if rgt_xn < self.rgt_yuzhi_1:
                    if (self.max_revn < revn_xn):
                        # if (self.max_revn<100):
                        self.max_revn = revn_xn

                        self.saver.save(sess,
                                        '/home/cosi/project/Integrated_RegNet/regretNet/experiments/record',
                                        global_step=iter)
                        #312_aaa3_LN3  312_g_8_333  402_d05_1

                        self.train_gen.save_data(iter)  # 数据再仔细研究论文，再议论一下50000次这个数据

                        self.max_iter = iter
                        self.max_rgt = rgt_xn

                        self.max_model_path = os.path.join(self.config.max_dir_name, 'model-' + str(iter))



            #iter > self.config.train.max_iter - 200000 150000
            # if (iter > self.config.train.max_iter - 100):
            if (iter > self.config.train.max_iter - 10):
                if rgt_xn < self.rgt_yuzhi_1:

                    metric_tot = np.zeros(len(self.metric_names_t))

                    for _ in range(self.config.test.num_batches):
                        AD, ORG, BID, VOL, graph_raw, graph_ranked, bundle_item_pair, ADV, _ = next(self.test_gen.gen_func)
                        # X, ADV, idxxx = next(self.val_gen.gen_func)
                        # print("val val val")
                        # print(idxxx)
                        # print("val val val x")
                        sess.run(self.assign_op_CTR, feed_dict={self.adv_init_CTR: ADV_CTR[:,:,:self.config.num_ad,:]})
                        for k in range(self.config.test.gd_iter):
                            sess.run(self.test_mis_step, feed_dict={self.ad: AD, self.org: ORG, self.bid: BID, self.vol:VOL, \
                                                                    self.graph_raw: graph_raw, self.graph_ranked:graph_ranked, \
                                                                    self.bundle_item_pair: bundle_item_pair})
                            sess.run(self.clip_op_test)
                        sess.run(self.reset_test_mis_opt)
                        metric_vals = sess.run(self.metrics_t, feed_dict={self.ad: AD, self.org: ORG, self.bid: BID, self.vol:VOL, \
                                                                          self.graph_raw: graph_raw, self.graph_ranked:graph_ranked, \
                                                                          self.bundle_item_pair: bundle_item_pair})
                        metric_tot += metric_vals

                    metric_tot = metric_tot / self.config.test.num_batches
                    fmt_vals = tuple([item for tup in zip(self.metric_names_t, metric_tot) for item in tup])
                    log_str = "TEST-%d" % (iter) + ", %s: %.6f" * len(
                        self.metric_names_t) % fmt_vals + " ,self.rgt_yuzhi：%.6f" % (
                                  self.rgt_yuzhi) + " ,rgt_xn：%.6f" % (rgt_xn) + " ,revn_xn：%.6f" % (revn_xn)
                    self.logger.info(log_str)

                    # self.max_revn_test = -100
                    # self.max_iter_test = -1
                    # self.max_rgt_test = -1
                    revn_xn1 = metric_tot[0]
                    rgt_xn1 = metric_tot[1]
                    ir_xn1 = metric_tot[2]
                    social_welfare_111 = metric_tot[3]
                    if rgt_xn1 < self.rgt_yuzhi:
                        if (self.max_revn_test < revn_xn1):
                            # if (self.max_revn_test < 100):
                            self.max_revn_test = revn_xn1
                            self.saver.save(sess, os.path.join(self.config.max_dir_name, 'model'), global_step=iter)
                            self.saver1.save(sess,
                                             '/home/cosi/project/Integrated_RegNet/regretNet/experiments/record1',
                                             global_step=iter)
                            #510_g_1_test  510_g_1_test_2 510_g_1_test_3
                            #510_d_1_test  510_d_1_test_2 510_g_1_test_3
                            #512_a_1_test_1         52777_a_1_test_1
                            #512_e_1_test_1
                            #512_g6_1_test_1  512_g8_1_test_1 512_g10_1_test_1
                            #512_d01_1_test_1 512_d026_1_test_1  512_d05_1_test_1
                            #512_N_1_test_1 512_LN_1_test_1
                            # 52777_a_1_test_7 第一次大规模运行跑到了7
                            # 52777_a_1_test_8

                            # tongyi
                            #61555_a_1_test_1 61555_a_1_test_3 61555_e_1_test_1  61555_n_1_test_3
                            #81666_a_1_test_1  81666_d01_1_test_1  81666_g6_1_test_1  81666_n_1_test_1
                            self.test_gen.save_data(iter)  # 数据再仔细研究论文，再议论一下50000次这个数据
                            self.max_iter_test = iter
                            self.max_rgt_test = rgt_xn1
                            self.max_ir_test = ir_xn1
                            self.max_social_welfare_test = social_welfare_111



            if (iter == self.config.train.max_iter):
                print("max_iter:")
                print(self.max_iter)
                print("max_rgt:")
                print(self.max_rgt)
                print("max_revn:")
                print(self.max_revn)

                print("max_iter_test:")
                print(self.max_iter_test)
                print("max_rgt_test:")
                print(self.max_rgt_test)
                print("max_revn_test:")
                print(self.max_revn_test)
                print("max_ir_test:")
                print(self.max_ir_test)
                print("max_social_welfare_test:")
                print(self.max_social_welfare_test)

            if (iter % self.config.val.print_iter) == 0:
                #Validation Set Stats
                metric_tot = np.zeros(len(self.metric_names))         
                for _ in range(self.config.val.num_batches):
                    AD, ORG, BID, VOL, graph_raw, graph_ranked, bundle_item_pair, ADV, _ = next(self.val_gen.gen_func)
                    # X, ADV, idxxx = next(self.val_gen.gen_func)
                    # print("val val val")
                    # print(idxxx)
                    # print("val val val x")
                    # print(X)
                    sess.run(self.assign_op, feed_dict = {self.adv_init: ADV})               
                    for k in range(self.config.val.gd_iter):
                        sess.run(self.val_mis_step, feed_dict = {self.ad: AD, self.org: ORG, self.bid: BID, self.vol:VOL, \
                                                                 self.graph_raw: graph_raw, self.graph_ranked:graph_ranked, \
                                                                 self.bundle_item_pair: bundle_item_pair})
                        sess.run(self.clip_op)
                    sess.run(self.reset_val_mis_opt)                                   
                    metric_vals = sess.run(self.metrics, feed_dict = {self.ad: AD, self.org: ORG, self.bid: BID, self.vol:VOL, \
                                                                      self.graph_raw: graph_raw, self.graph_ranked:graph_ranked, \
                                                                      self.bundle_item_pair: bundle_item_pair})
                    metric_tot += metric_vals
                    
                metric_tot = metric_tot/self.config.val.num_batches
                fmt_vals = tuple([ item for tup in zip(self.metric_names, metric_tot) for item in tup ])
                log_str = "VAL-%d"%(iter) + ", %s: %.6f"*len(self.metric_names)%fmt_vals
                self.logger.info(log_str)



    def test(self, generator):
        """
        Runs test
        """
        
        # Init generators
        self.test_gen = generator

        iter = self.config.test.restore_iter
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        # model_path = os.path.join(self.config.dir_name,'model-' + str(iter))
        model_path = ''

        self.saver.restore(sess, model_path)


        print("Model restored.\n")

        #Test-set Stats
        time_elapsed = 0
            
        metric_tot = np.zeros(len(self.metric_names))

        if self.config.test.save_output:
            assert(hasattr(generator, "X")), "save_output option only allowed when config.test.data = Fixed or when X is passed as an argument to the generator"
            alloc_tst = np.zeros(self.test_gen.X.shape)
            pay_tst = np.zeros(self.test_gen.X.shape[:-1])
                    
        for i in range(self.config.test.num_batches):
            tic = time.time()
            AD, ORG, BID, VOL, graph_raw, graph_ranked, bundle_item_pair, ADV, perm = next(self.test_gen.gen_func)
            sess.run(self.assign_op, feed_dict = {self.adv_init: ADV})
                    
            for k in range(self.config.test.gd_iter):
                sess.run(self.test_mis_step, feed_dict = {self.ad: AD, self.org: ORG, self.bid: BID, self.vol:VOL, \
                                                          self.graph_raw: graph_raw, self.graph_ranked:graph_ranked, \
                                                          self.bundle_item_pair: bundle_item_pair})
                sess.run(self.clip_op)

            sess.run(self.reset_test_mis_opt)        
                
            metric_vals = sess.run(self.metrics, feed_dict = {self.ad: AD, self.org: ORG, self.bid: BID, self.vol:VOL, \
                                                              self.graph_raw: graph_raw, self.graph_ranked:graph_ranked, \
                                                              self.bundle_item_pair: bundle_item_pair})
            
            if self.config.test.save_output:
                A, P = sess.run([self.alloc, self.pay], feed_dict = {self.ad: AD, self.org: ORG, self.bid: BID, self.vol:VOL, \
                                                                     self.graph_raw: graph_raw, self.graph_ranked:graph_ranked, \
                                                                     self.bundle_item_pair: bundle_item_pair})
                alloc_tst[perm, :, :] = A
                pay_tst[perm, :] = P
                    
            metric_tot += metric_vals
            toc = time.time()
            time_elapsed += (toc - tic)

            fmt_vals = tuple([ item for tup in zip(self.metric_names, metric_vals) for item in tup ])
            log_str = "TEST BATCH-%d: t = %.4f"%(i, time_elapsed) + ", %s: %.6f"*len(self.metric_names)%fmt_vals
            self.logger.info(log_str)
        
        metric_tot = metric_tot/self.config.test.num_batches
        fmt_vals = tuple([ item for tup in zip(self.metric_names, metric_tot) for item in tup ])
        log_str = "TEST ALL-%d: t = %.4f"%(iter, time_elapsed) + ", %s: %.6f"*len(self.metric_names)%fmt_vals
        self.logger.info(log_str)
            
        if self.config.test.save_output:
            np.save(os.path.join(self.config.dir_name, 'alloc_tst_' + str(iter)), alloc_tst)
            np.save(os.path.join(self.config.dir_name, 'pay_tst_' + str(iter)), pay_tst)
